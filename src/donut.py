"""Modelo Donut ajustado para facturas FEL (encoder Swin + decoder BART, sin OCR).

El modelo genera una secuencia como
    <s_fel><s_nit_emisor>4809526-5</s_nit_emisor><s_nit_receptor>CF</s_nit_receptor>...</s>
que aquí se convierte en un dict con los campos de src/schema.py. Además de los valores,
devuelve la confianza de cada campo (la probabilidad mínima de sus tokens), que se usa para
marcar campos dudosos y mandarlos a revisión humana.

Entrenamiento: train/train_donut.py.
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path

from . import config
from .schema import FIELDS

TASK_TOKEN = "<s_fel>"
FIELD_TOKENS = [t for f in FIELDS for t in (f"<s_{f}>", f"</s_{f}>")]
MAX_LENGTH = 160


def to_sequence(gt: dict) -> str:
    """{campo: valor} -> '<s_campo>valor</s_campo>...'. Un valor nulo queda vacío."""
    return "".join(f"<s_{f}>{'' if gt.get(f) is None else gt[f]}</s_{f}>" for f in FIELDS)


def from_sequence(seq: str) -> dict:
    out = {}
    for f in FIELDS:
        m = re.search(rf"<s_{f}>(.*?)</s_{f}>", seq, flags=re.S)
        val = m.group(1).strip() if m else ""
        out[f] = val or None
    return out


@dataclass
class DonutOutput:
    fields: dict
    confidence: dict  # campo -> probabilidad mínima de sus tokens (0 a 1)
    sequence: str
    latency_s: float


class DonutFEL:
    def __init__(self, model_dir: str | Path | None = None, device: str | None = None):
        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        self.torch = torch
        model_dir = Path(model_dir or config.DONUT_MODEL_DIR)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"No hay un modelo Donut en {model_dir}. Entrénalo con train/train_donut.py "
                "o pon la ruta correcta en DONUT_MODEL_DIR.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = DonutProcessor.from_pretrained(model_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir, dtype=dtype).to(self.device).eval()
        tok = self.processor.tokenizer
        self.start_id = tok.convert_tokens_to_ids(TASK_TOKEN)
        self.open_ids = {tok.convert_tokens_to_ids(f"<s_{f}>"): f for f in FIELDS}
        self.close_ids = {tok.convert_tokens_to_ids(f"</s_{f}>") for f in FIELDS}
        self.name = model_dir.name

    def predict_images(self, images: list) -> list[DonutOutput]:
        """images: lista de PIL.Image en RGB."""
        torch = self.torch
        t0 = time.perf_counter()
        pix = self.processor(images, return_tensors="pt").pixel_values.to(self.device, self.model.dtype)
        start = torch.full((len(images), 1), self.start_id, device=self.device)
        tok = self.processor.tokenizer
        with torch.inference_mode():
            out = self.model.generate(
                pix, decoder_input_ids=start, max_length=MAX_LENGTH, num_beams=1, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                bad_words_ids=[[tok.unk_token_id]], output_scores=True, return_dict_in_generate=True)
        latency = (time.perf_counter() - t0) / len(images)
        # Probabilidad del token elegido en cada paso: (batch, pasos)
        probs = torch.stack([s.float().softmax(-1) for s in out.scores], 1)
        gen = out.sequences[:, 1:]
        chosen = probs.gather(-1, gen[:, : probs.shape[1], None]).squeeze(-1).cpu().tolist()

        results = []
        for i, ids in enumerate(gen.cpu().tolist()):
            conf, field = {}, None
            for tid, p in zip(ids, chosen[i]):
                if tid in self.open_ids:
                    field = self.open_ids[tid]
                    conf[field] = p
                elif tid in self.close_ids:
                    if field:
                        conf[field] = min(conf[field], p)
                    field = None
                elif field:
                    conf[field] = min(conf[field], p)
                if tid == tok.eos_token_id:
                    break
            seq = tok.decode(ids, skip_special_tokens=False)
            seq = seq.replace(tok.eos_token, "").replace(tok.pad_token, "")
            fields = from_sequence(seq)
            results.append(DonutOutput(fields, {f: round(conf.get(f, 0.0), 4) for f in FIELDS},
                                       seq, latency))
        return results


_MODEL = None


def get_model() -> DonutFEL:
    """Carga el modelo una sola vez por proceso."""
    global _MODEL
    if _MODEL is None:
        _MODEL = DonutFEL()
    return _MODEL
