"""Métricas técnicas y de negocio a partir de las predicciones por factura."""

import statistics

from . import config
from . import normalize as N
from .schema import ERROR_COST_Q, EVAL_FIELDS


def score_record(pred: dict, gold: dict) -> dict:
    """Por campo: tp / fp / fn y si quedó correcto."""
    out = {}
    for f in EVAL_FIELDS:
        p, g = N.NORMALIZERS[f](pred.get(f)), N.NORMALIZERS[f](gold.get(f))
        ok = N.equal(f, p, g)
        out[f] = {
            "correcto": ok,
            "tp": int(p is not None and ok),
            "fp": int(p is not None and not ok),
            "fn": int(g is not None and not ok),
            "inventado": int(p is not None and g is None),
        }
    return out


def _f1(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def business(rows: list[dict], policy: str) -> dict:
    """Ahorro neto por cada 1,000 facturas frente a digitarlas a mano.

    policy: "ninguna" (acepta todo), "errores" (revisa si falla una validación)
            o "estricta" (revisa también con advertencias o campos ilegibles).
    """
    wage_min = config.HOURLY_WAGE_GTQ / 60
    n = len(rows)
    review = escaped_cost = escaped_docs = api = 0.0
    for r in rows:
        api += r["costo_usd"] * config.USD_TO_GTQ
        flagged = {"ninguna": False, "errores": r["revision_errores"],
                   "estricta": r["revision_estricta"]}[policy]
        if flagged:
            review += config.REVIEW_MINUTES_PER_INVOICE * wage_min
        else:
            c = sum(ERROR_COST_Q[f] for f in EVAL_FIELDS if not r["score"][f]["correcto"])
            escaped_cost += c
            escaped_docs += c > 0
    manual = n * config.MANUAL_MINUTES_PER_INVOICE * wage_min
    k = 1000 / n
    return {
        "politica": policy,
        "pct_a_revision": sum(r[f"revision_{policy}"] for r in rows) / n if policy != "ninguna" else 0.0,
        "pct_errores_escapados": escaped_docs / n,
        "costo_manual_Q_1000": manual * k,
        "costo_sistema_Q_1000": (api + review + escaped_cost) * k,
        "ahorro_neto_Q_1000": (manual - api - review - escaped_cost) * k,
        "horas_ahorradas_1000": (manual - review) * k / config.HOURLY_WAGE_GTQ,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    tp = sum(r["score"][f]["tp"] for r in rows for f in EVAL_FIELDS)
    fp = sum(r["score"][f]["fp"] for r in rows for f in EVAL_FIELDS)
    fn = sum(r["score"][f]["fn"] for r in rows for f in EVAL_FIELDS)
    prec, rec, f1 = _f1(tp, fp, fn)
    lat = [r["latencia_s"] for r in rows]
    wrong_docs = [not all(r["score"][f]["correcto"] for f in EVAL_FIELDS) for r in rows]
    flagged = [r["revision_errores"] for r in rows]
    # ¿Las validaciones detectan las facturas con errores?
    det_tp = sum(w and fl for w, fl in zip(wrong_docs, flagged))
    det_prec = det_tp / sum(flagged) if sum(flagged) else 0.0
    det_rec = det_tp / sum(wrong_docs) if sum(wrong_docs) else 1.0

    s = {
        "n": n,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "exactitud_documento": 1 - sum(wrong_docs) / n,
        "tasa_inventados": sum(r["score"][f]["inventado"] for r in rows for f in EVAL_FIELDS) / (n * len(EVAL_FIELDS)),
        "errores_llm": sum(bool(r["error"]) for r in rows),
        "latencia_media_s": statistics.mean(lat),
        "latencia_p95_s": sorted(lat)[max(0, int(0.95 * n) - 1)],
        "tokens_entrada_media": statistics.mean(r["tokens_in"] for r in rows),
        "tokens_salida_media": statistics.mean(r["tokens_out"] for r in rows),
        "costo_usd_por_factura": statistics.mean(r["costo_usd"] for r in rows),
        "validacion_precision": det_prec,
        "validacion_recall": det_rec,
    }
    for f in EVAL_FIELDS:
        s[f"acc_{f}"] = sum(r["score"][f]["correcto"] for r in rows) / n
    for pol in ("ninguna", "errores", "estricta"):
        b = business(rows, pol)
        s[f"ahorro_Q_1000_{pol}"] = b["ahorro_neto_Q_1000"]
        s[f"pct_revision_{pol}"] = b["pct_a_revision"]
        s[f"pct_escapados_{pol}"] = b["pct_errores_escapados"]
    return s
