"""Curva de valor del modelo Donut según el umbral de confianza.

Para cada umbral t, una factura va a revisión humana si falla alguna validación FEL o si
algún campo tiene confianza < t. Revisar cuesta minutos de un digitador; no revisar deja
escapar los errores, que cuestan según el campo (src/schema.py). Se busca el t que maximiza
el ahorro neto frente a digitar todo a mano, y se repite moviendo los supuestos ±20 % y ±50 %
(análisis de sensibilidad).

    python -m eval.threshold_curve --synth data/synth/val      # validación sintética
    python -m eval.threshold_curve --gt eval/ground_truth.csv  # fotos reales (split=test)

Salida en results/umbral_<modelo>_<conjunto>/: predicciones.json, curva.csv, sensibilidad.csv y curva.png.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.extractors import _clean  # noqa: E402
from src.metrics import score_record  # noqa: E402
from src.schema import ERROR_COST_Q, EVAL_FIELDS  # noqa: E402
from src.validate import needs_review, validate  # noqa: E402

THRESHOLDS = np.round(np.concatenate([np.linspace(0, 0.9, 19), np.linspace(0.91, 1.0, 10)]), 3)


def load_records(a):
    if a.synth:
        d = Path(a.synth)
        recs = [json.loads(line) for line in (d / "metadata.jsonl").open(encoding="utf-8")]
        return ([(d / r["file"], {**r["gt"], "_formato": r["meta"]["layout"]}, r["file"]) for r in recs],
                f"{d.parent.name}_{d.name}")
    df = pd.read_csv(a.gt, dtype=str, keep_default_na=False)
    df = df[(df["split"] == a.split) & ~df["notas"].str.contains("borrar esta fila", na=False)]
    base = Path(a.gt).parent
    return [(base / r["archivo_imagen"], r.to_dict(), r["id"]) for _, r in df.iterrows()], f"real_{a.split}"


def predict(records, cache: Path, model_dir: Path, constrained: bool):
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    from src.donut import DonutFEL

    model = DonutFEL(model_dir, constrained=constrained)
    preds = []
    for i in range(0, len(records), 8):
        chunk = records[i:i + 8]
        outs = model.predict_images([Image.open(p).convert("RGB") for p, _, _ in chunk])
        for (_, _, rid), o in zip(chunk, outs):
            preds.append({"id": rid, "fields": o.fields, "confidence": o.confidence, "latency_s": o.latency_s})
        print(f"  {min(i + 8, len(records))}/{len(records)}", flush=True)
    cache.write_text(json.dumps(preds, ensure_ascii=False, indent=1), encoding="utf-8")
    return preds


def value_per_1000(rows, t, cost_mult=1.0, wage_mult=1.0, manual_mult=1.0):
    """Ahorro neto (Q por 1,000 facturas) con umbral t y supuestos escalados."""
    wage_min = config.HOURLY_WAGE_GTQ * wage_mult / 60
    review = escaped = n_rev = n_esc = 0.0
    for r in rows:
        if r["falla_validacion"] or r["conf_min"] < t:
            review += config.REVIEW_MINUTES_PER_INVOICE * wage_min
            n_rev += 1
        elif r["costo_errores"] > 0:
            escaped += r["costo_errores"] * cost_mult
            n_esc += 1
    n = len(rows)
    manual = n * config.MANUAL_MINUTES_PER_INVOICE * manual_mult * wage_min
    k = 1000 / n
    return {"ahorro_neto_Q_1000": (manual - review - escaped) * k, "pct_revision": n_rev / n,
            "pct_errores_escapados": n_esc / n, "costo_revision_Q_1000": review * k,
            "costo_errores_Q_1000": escaped * k, "costo_manual_Q_1000": manual * k}


def main(a):
    records, name = load_records(a)
    model_dir = Path(a.model or config.DONUT_MODEL_DIR)
    suffix = "_restringido" if a.constrained else ""
    out = ROOT / "results" / f"umbral_{model_dir.name}{suffix}_{name}"
    out.mkdir(parents=True, exist_ok=True)
    preds = predict(records, out / "predicciones.json", model_dir, a.constrained)

    rows, scores = [], []
    for (_, gt, _), p in zip(records, preds):
        fields, _ = _clean(p["fields"])  # misma normalización y autocorrección que la app
        sc = score_record(fields, gt)
        scores.append((gt.get("_formato", "real"), sc))
        rows.append({
            "falla_validacion": needs_review(validate(fields)),
            "conf_min": min(p["confidence"][f] for f in EVAL_FIELDS),
            "costo_errores": sum(ERROR_COST_Q[f] for f in EVAL_FIELDS if not sc[f]["correcto"]),
        })

    def accuracy(subset):
        n = len(subset)
        acc = {f"acc_{f}": round(sum(sc[f]["correcto"] for _, sc in subset) / n, 4) for f in EVAL_FIELDS}
        return {"n": n, "acc_campos": round(sum(acc.values()) / len(acc), 4),
                "exactitud_documento": round(sum(all(sc[f]["correcto"] for f in EVAL_FIELDS)
                                                 for _, sc in subset) / n, 4), **acc}

    metricas = {"modelo": model_dir.name, "restringido": a.constrained, "conjunto": name,
                "latencia_media_s": round(float(np.mean([p["latency_s"] for p in preds])), 3),
                "todas": accuracy(scores)}
    for fmt in sorted({f for f, _ in scores}):
        metricas[fmt] = accuracy([x for x in scores if x[0] == fmt])

    curve = pd.DataFrame([{"umbral": t, **value_per_1000(rows, t)} for t in THRESHOLDS])
    curve.to_csv(out / "curva.csv", index=False)
    best = curve.loc[curve["ahorro_neto_Q_1000"].idxmax()]

    sens = []
    for label, kw in [("base", {}),
                      ("costo errores -50%", {"cost_mult": 0.5}), ("costo errores +50%", {"cost_mult": 1.5}),
                      ("costo errores -20%", {"cost_mult": 0.8}), ("costo errores +20%", {"cost_mult": 1.2}),
                      ("salario -20%", {"wage_mult": 0.8}), ("salario +20%", {"wage_mult": 1.2}),
                      ("min. digitación -50%", {"manual_mult": 0.5}), ("min. digitación +50%", {"manual_mult": 1.5})]:
        vals = [value_per_1000(rows, t, **kw)["ahorro_neto_Q_1000"] for t in THRESHOLDS]
        i = int(np.argmax(vals))
        sens.append({"escenario": label, "umbral_optimo": THRESHOLDS[i], "ahorro_optimo_Q_1000": round(vals[i], 1),
                     "ahorro_con_umbral_base_Q_1000": round(vals[list(THRESHOLDS).index(best["umbral"])], 1)})
    sens = pd.DataFrame(sens)
    sens.to_csv(out / "sensibilidad.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(curve["umbral"], curve["ahorro_neto_Q_1000"], "o-", label="base")
    for label, kw in [("costo errores +50%", {"cost_mult": 1.5}), ("costo errores -50%", {"cost_mult": 0.5})]:
        ax[0].plot(THRESHOLDS, [value_per_1000(rows, t, **kw)["ahorro_neto_Q_1000"] for t in THRESHOLDS],
                   "--", label=label)
    ax[0].axvline(best["umbral"], color="gray", ls=":")
    ax[0].set(title=f"Ahorro neto por 1,000 facturas (óptimo: umbral {best['umbral']:.2f})",
              xlabel="umbral de confianza", ylabel="Q")
    ax[0].legend()
    ax[1].plot(curve["umbral"], curve["pct_revision"], "o-", label="% a revisión humana")
    ax[1].plot(curve["umbral"], curve["pct_errores_escapados"], "s-", label="% con errores que se escapan")
    ax[1].set(title="Punto de operación", xlabel="umbral de confianza", ylim=(0, 1))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "curva.png", dpi=130)

    metricas["umbral_optimo"] = float(best["umbral"])
    metricas["en_umbral_optimo"] = {k: round(float(v), 4) for k, v in best.items()}
    metricas["sin_umbral_solo_validaciones"] = {k: round(float(v), 4) for k, v in curve.iloc[0].items()}
    (out / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(metricas["todas"], ensure_ascii=False))

    print(f"\nn={len(rows)}  latencia media {np.mean([p['latency_s'] for p in preds]):.2f} s")
    print(curve.round(3).to_string(index=False))
    print(f"\nUmbral óptimo: {best['umbral']:.2f} -> ahorro Q{best['ahorro_neto_Q_1000']:.0f} por 1,000 facturas, "
          f"{best['pct_revision']:.0%} a revisión, {best['pct_errores_escapados']:.0%} con errores escapados")
    print("\n" + sens.to_string(index=False))
    print(f"\nResultados en {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--synth", help="carpeta con metadata.jsonl (p. ej. data/synth/val)")
    g.add_argument("--gt", help="CSV de ground truth con fotos reales")
    p.add_argument("--split", default="test")
    p.add_argument("--model", default="", help="carpeta del modelo (por defecto DONUT_MODEL_DIR)")
    p.add_argument("--constrained", action="store_true", help="decodificación restringida al formato")
    main(p.parse_args())
