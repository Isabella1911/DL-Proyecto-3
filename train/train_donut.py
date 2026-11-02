"""Fine-tuning de Donut para extraer los campos de facturas FEL.

Parte de naver-clova-ix/donut-base-finetuned-cord-v2 (ya sabe leer recibos) y lo ajusta
para generar la secuencia de src/donut.py. Por defecto entrena con los datos sintéticos de
train/synth_fel.py; con --extra se agregan fotos reales anotadas (eval/ground_truth.csv).

    python -m train.train_donut --epochs 4 --out models/donut-fel
    python -m train.train_donut --smoke          # prueba rápida de que todo corre
    # Iteración 2: parte del modelo anterior y agrega otro lote sintético
    python -m train.train_donut --base models/donut-fel --more-train data/synth2/train \
        --epochs 2 --lr 2e-5 --out models/donut-fel-it2

Guarda en --out el mejor modelo según la exactitud de campos en validación, y en
--out/logs: steps.csv (loss por paso), epochs.csv (métricas por época), curvas.png y
train_summary.json (tiempo, GPU, memoria máxima: costo de entrenamiento para el informe).
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

# Evita fragmentación de memoria: con 8 GB de VRAM el margen es chico.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from PIL import Image
from torch.utils.data import DataLoader, Dataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.donut import FIELD_TOKENS, MAX_LENGTH, TASK_TOKEN, DonutFEL, to_sequence  # noqa: E402
from src.metrics import score_record  # noqa: E402
from src.schema import EVAL_FIELDS  # noqa: E402

BASE_MODEL = "naver-clova-ix/donut-base-finetuned-cord-v2"


class FELDataset(Dataset):
    def __init__(self, records, processor, train: bool):
        self.records, self.processor, self.train = records, processor, train
        self.tok = processor.tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        path, gt = self.records[i]
        img = Image.open(path).convert("RGB")
        pix = self.processor(img, return_tensors="pt").pixel_values[0]
        ids = self.tok(to_sequence(gt), add_special_tokens=False, max_length=MAX_LENGTH - 1,
                       truncation=True).input_ids + [self.tok.eos_token_id]
        return pix, torch.tensor(ids)


def collate(batch, pad_id):
    pix = torch.stack([b[0] for b in batch])
    n = max(len(b[1]) for b in batch)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    for i, (_, ids) in enumerate(batch):
        labels[i, : len(ids)] = ids
    return pix, labels


def load_synth(split_dir: Path, limit: int = 0):
    recs = []
    with (split_dir / "metadata.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            recs.append((split_dir / r["file"], r["gt"]))
    return recs[:limit] if limit else recs


def load_real(csv_path: Path, splits=("train",)):
    """Fotos reales anotadas (mismo formato que eval/ground_truth.csv)."""
    import pandas as pd

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df = df[df["split"].isin(splits)]
    fields = ["nit_emisor", "nit_receptor", "nombre_emisor", "fecha", "serie", "numero_dte",
              "numero_autorizacion", "iva", "total"]
    return [(csv_path.parent / r["archivo_imagen"], {f: (r.get(f) or None) for f in fields})
            for _, r in df.iterrows()]


@torch.no_grad()
def evaluate(model, processor, loader, records, device, max_gen: int, gen_batch: int = 8):
    """Loss de validación + exactitud por campo generando con el modelo."""
    model.eval()
    losses = []
    for pix, labels in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses.append(model(pixel_values=pix.to(device), labels=labels.to(device)).loss.item())

    wrapper = DonutFEL.__new__(DonutFEL)  # reusa la inferencia de src/donut.py sin recargar
    wrapper.torch, wrapper.device, wrapper.processor = torch, device, processor
    wrapper.model, wrapper.name = model, "train"
    tok = processor.tokenizer
    wrapper.start_id = tok.convert_tokens_to_ids(TASK_TOKEN)
    wrapper.open_ids = {tok.convert_tokens_to_ids(t): t[3:-1] for t in FIELD_TOKENS if not t.startswith("</")}
    wrapper.close_ids = {tok.convert_tokens_to_ids(t) for t in FIELD_TOKENS if t.startswith("</")}

    sub = records[:max_gen]
    correct = {f: 0 for f in EVAL_FIELDS}
    docs_ok = 0
    for i in range(0, len(sub), gen_batch):
        chunk = sub[i:i + gen_batch]
        imgs = [Image.open(p).convert("RGB") for p, _ in chunk]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = wrapper.predict_images(imgs)
        for (_, gt), o in zip(chunk, outs):
            sc = score_record(o.fields, gt)
            for f in EVAL_FIELDS:
                correct[f] += sc[f]["correcto"]
            docs_ok += all(sc[f]["correcto"] for f in EVAL_FIELDS)
    n = len(sub)
    acc = {f"acc_{f}": correct[f] / n for f in EVAL_FIELDS}
    model.train()
    return {"val_loss": sum(losses) / len(losses), "acc_campos": sum(acc.values()) / len(acc),
            "exactitud_documento": docs_ok / n, **acc}


def plot(log_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    steps = pd.read_csv(log_dir / "steps.csv")
    ep = pd.read_csv(log_dir / "epochs.csv")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(steps["step"], steps["loss"], alpha=0.3, label="train (por paso)")
    ax[0].plot(steps["step"], steps["loss"].rolling(25, min_periods=1).mean(), label="train (media móvil)")
    ax[0].plot(ep["step"], ep["val_loss"], "o-", label="validación")
    ax[0].set(title="Loss (entropía cruzada)", xlabel="paso de optimización", yscale="log")
    ax[0].legend()
    ax[1].plot(ep["epoca"], ep["acc_campos"], "o-", label="exactitud media por campo")
    ax[1].plot(ep["epoca"], ep["exactitud_documento"], "s-", label="documento completo correcto")
    ax[1].set(title="Validación", xlabel="época", ylim=(0, 1.02))
    ax[1].legend()
    for c in [c for c in ep if c.startswith("acc_") and c != "acc_campos"]:
        ax[2].plot(ep["epoca"], ep[c], "o-", label=c[4:])
    ax[2].set(title="Exactitud por campo (validación)", xlabel="época", ylim=(0, 1.02))
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(log_dir / "curvas.png", dpi=130)
    plt.close(fig)


def main(a):
    from transformers import DonutProcessor, VisionEncoderDecoderModel, get_cosine_schedule_with_warmup

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda"
    out = Path(a.out)
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    processor = DonutProcessor.from_pretrained(a.base)
    processor.image_processor.size = {"height": a.height, "width": a.width}
    processor.image_processor.do_align_long_axis = False
    tok = processor.tokenizer
    tok.add_special_tokens({"additional_special_tokens": [TASK_TOKEN] + FIELD_TOKENS})

    # float32 explícito: el modelo guardado está en bf16 y entrenar en bf16 puro es inestable
    model = VisionEncoderDecoderModel.from_pretrained(a.base, dtype=torch.float32)
    model.decoder.resize_token_embeddings(len(tok))
    model.config.encoder.image_size = [a.height, a.width]
    model.config.decoder_start_token_id = tok.convert_tokens_to_ids(TASK_TOKEN)
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    gc = model.generation_config
    gc.decoder_start_token_id, gc.pad_token_id, gc.eos_token_id = (
        model.config.decoder_start_token_id, tok.pad_token_id, tok.eos_token_id)
    gc.max_length = MAX_LENGTH
    if a.grad_ckpt:
        model.encoder.gradient_checkpointing_enable()
    model.to(device).train()

    limit = 64 if a.smoke else 0
    train_recs = load_synth(Path(a.data) / "train", limit)
    for d in a.more_train:
        train_recs += load_synth(Path(d), limit)
    val_recs = load_synth(Path(a.data) / "val", 16 if a.smoke else 0)
    if a.extra:
        real = load_real(Path(a.extra))
        train_recs += real * a.extra_repeat  # sobremuestreo: hay pocas fotos reales
        print(f"+ {len(real)} fotos reales (x{a.extra_repeat})")
    print(f"train={len(train_recs)} val={len(val_recs)}  imagen={a.height}x{a.width}")

    coll = partial(collate, pad_id=tok.pad_token_id)  # partial y no lambda: funciona en Windows
    train_dl = DataLoader(FELDataset(train_recs, processor, True), batch_size=a.batch, shuffle=True,
                          num_workers=a.workers, collate_fn=coll, drop_last=True, persistent_workers=a.workers > 0)
    val_dl = DataLoader(FELDataset(val_recs, processor, False), batch_size=a.batch, num_workers=a.workers,
                        collate_fn=coll)

    steps_per_epoch = len(train_dl) // a.accum
    total = (1 if a.smoke else a.epochs) * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, min(a.warmup, total // 10), total)

    steps_f = (log_dir / "steps.csv").open("w", newline="")
    sw = csv.writer(steps_f)
    sw.writerow(["step", "epoca", "loss", "lr", "seg"])
    epochs_rows, best, step, t_start = [], -1.0, 0, time.time()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, (1 if a.smoke else a.epochs) + 1):
        acc_loss, t_ep = 0.0, time.time()
        for i, (pix, labels) in enumerate(train_dl, 1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(pixel_values=pix.to(device, non_blocking=True),
                             labels=labels.to(device, non_blocking=True)).loss / a.accum
            loss.backward()
            acc_loss += loss.item()
            if i % a.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            sw.writerow([step, epoch, round(acc_loss, 5), sched.get_last_lr()[0], round(time.time() - t_start, 1)])
            if step % 25 == 0 or a.smoke:
                steps_f.flush()
                el = time.time() - t_ep
                eta = el / (step - (epoch - 1) * steps_per_epoch) * (total - step)
                print(f"ep {epoch} paso {step}/{total} loss {acc_loss:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"| {el / 60:.1f} min en la época, faltan ~{eta / 60:.0f} min "
                      f"| VRAM máx {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            acc_loss = 0.0

        t_val = time.time()
        m = evaluate(model, processor, val_dl, val_recs, device, a.val_gen, a.gen_batch)
        row = {"epoca": epoch, "step": step, "min_entrenamiento": round((time.time() - t_ep) / 60, 2),
               "min_validacion": round((time.time() - t_val) / 60, 2), **{k: round(v, 4) for k, v in m.items()}}
        epochs_rows.append(row)
        with (log_dir / "epochs.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            w.writeheader()
            w.writerows(epochs_rows)
        plot(log_dir)
        print(f"== época {epoch}: val_loss {m['val_loss']:.4f}  exactitud campos {m['acc_campos']:.3f}  "
              f"documento {m['exactitud_documento']:.3f}", flush=True)
        if m["acc_campos"] > best:
            best = m["acc_campos"]
            model.save_pretrained(out, safe_serialization=True)
            processor.save_pretrained(out)
            print(f"   guardado (mejor hasta ahora) en {out}", flush=True)

    steps_f.close()
    # El mejor modelo se guarda en bfloat16: pesa la mitad (~400 MB) y en CPU se carga en float32.
    best_model = VisionEncoderDecoderModel.from_pretrained(out, dtype=torch.bfloat16)
    best_model.save_pretrained(out, safe_serialization=True)
    hours = (time.time() - t_start) / 3600
    summary = {
        "modelo_base": a.base, "gpu": torch.cuda.get_device_name(), "horas": round(hours, 3),
        "vram_max_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "epocas": len(epochs_rows),
        "pasos": step, "batch_efectivo": a.batch * a.accum, "lr": a.lr, "imagen": [a.height, a.width],
        "n_train": len(train_recs), "n_val": len(val_recs), "mejor_acc_campos": best,
        "parametros_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
    }
    (log_dir / "train_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--data", default=str(ROOT / "data" / "synth"))
    p.add_argument("--more-train", nargs="*", default=[], help="más carpetas con metadata.jsonl")
    p.add_argument("--extra", default="", help="CSV con fotos reales anotadas (split=train)")
    p.add_argument("--extra-repeat", type=int, default=10)
    p.add_argument("--out", default=str(ROOT / "models" / "donut-fel"))
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--height", type=int, default=1280)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-gen", type=int, default=300, help="facturas de validación a generar por época")
    p.add_argument("--gen-batch", type=int, default=8, help="lote de generación en validación (menos = menos VRAM)")
    p.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    main(p.parse_args())
