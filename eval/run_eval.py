"""Evalúa los extractores contra eval/ground_truth.csv.

Ejemplos:
    python -m eval.run_eval --tag it1_baseline --methods baseline
    python -m eval.run_eval --tag it2_vision_v1 --methods vision --prompt v1_zero_shot
    python -m eval.run_eval --tag it3_vision_v2 --methods vision ocr_llm --prompt v2_fel_rules
    python -m eval.run_eval --tag it4_vision_fewshot --methods vision --fewshot 2

Cada corrida agrega una fila por método a results/iterations.csv (la tabla de la sección 2.5)
y guarda las respuestas en results/<tag>/ para no volver a pagar la API si se repite.
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extractors import Extraction, extract  # noqa: E402
from src.metrics import score_record, summarize  # noqa: E402
from src.schema import EVAL_FIELDS, FIELDS  # noqa: E402
from src.validate import needs_review, validate  # noqa: E402

GT = ROOT / "eval" / "ground_truth.csv"
RESULTS = ROOT / "results"


def load_gt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[~df["notas"].str.contains("borrar esta fila", na=False)]
    if "split" not in df.columns:
        df["split"] = "test"
    return df


def fewshot_examples(df: pd.DataFrame, n: int):
    ex = []
    for _, r in df[df["split"] == "fewshot"].head(n).iterrows():
        ans = {f: (r.get(f) or None) for f in FIELDS}
        ex.append(((ROOT / "eval" / r["archivo_imagen"]).read_bytes(), ans))
    return ex


def run(args):
    df = load_gt(Path(args.gt))
    test = df[df["split"] == args.split]
    if args.limit:
        test = test.head(args.limit)
    if test.empty:
        sys.exit(f"No hay filas con split={args.split} en {args.gt}")
    shots = fewshot_examples(df, args.fewshot) if args.fewshot else None

    out_dir = RESULTS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for method in args.methods:
        rows, detail, models = [], [], set()
        for _, g in test.iterrows():
            cache = out_dir / f"{method}__{g['id']}.json"
            if cache.exists() and not args.no_cache:
                ex = Extraction(**json.loads(cache.read_text(encoding="utf-8")))
            else:
                img = (ROOT / "eval" / g["archivo_imagen"]).read_bytes()
                try:
                    ex = extract(method, img, prompt=args.prompt, model=args.model,
                                 provider=args.provider, fewshot=shots)
                except Exception as e:  # p. ej. Tesseract no instalado
                    sys.exit(f"{method} falló en {g['id']}: {type(e).__name__}: {e}")
                if not ex.error:  # los errores (red, cuota) se reintentan en la próxima corrida
                    cache.write_text(json.dumps(asdict(ex), ensure_ascii=False, indent=1), encoding="utf-8")
            models.add(ex.model)
            issues = validate(ex.fields)
            row = {
                "id": g["id"], "score": score_record(ex.fields, g.to_dict()),
                "latencia_s": ex.latency_s, "tokens_in": ex.tokens_in, "tokens_out": ex.tokens_out,
                "costo_usd": ex.cost_usd, "error": ex.error,
                "revision_errores": needs_review(issues),
                "revision_estricta": needs_review(issues, strict=True) or bool(ex.ilegibles),
            }
            rows.append(row)
            d = {"id": g["id"], "metodo": method, "comercio": g.get("comercio"),
                 "calidad_foto": g.get("calidad_foto"), "error": ex.error,
                 "revision": row["revision_errores"],
                 "validaciones": "; ".join(f"{i['campo']}:{i['regla']}" for i in issues)}
            for f in EVAL_FIELDS:
                d[f"{f}_pred"] = ex.fields.get(f)
                d[f"{f}_gold"] = g.get(f)
                d[f"{f}_ok"] = row["score"][f]["correcto"]
            detail.append(d)
            print(f"  {method:9s} {g['id']}: {sum(d[f'{f}_ok'] for f in EVAL_FIELDS)}/{len(EVAL_FIELDS)}"
                  + (f"  ERROR {ex.error}" if ex.error else ""), flush=True)

        pd.DataFrame(detail).to_csv(out_dir / f"detalle_{method}.csv", index=False, encoding="utf-8-sig")
        s = summarize(rows)
        model = "-" if method == "baseline" else ",".join(sorted(m for m in models if m)) or "-"
        prompt = "-" if method == "baseline" else args.prompt + (f"+{args.fewshot}shot" if args.fewshot else "")
        summaries.append({"fecha": datetime.now().isoformat(timespec="minutes"), "iteracion": args.tag,
                          "metodo": method, "modelo": model, "prompt": prompt, "split": args.split,
                          "notas": args.notes, **s})

    it = RESULTS / "iterations.csv"
    new = not it.exists()
    with it.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        if new:
            w.writeheader()
        w.writerows(summaries)

    cols = ["metodo", "n", "f1", "exactitud_documento", "acc_nit_emisor", "acc_total",
            "latencia_media_s", "costo_usd_por_factura", "pct_revision_errores", "ahorro_Q_1000_errores"]
    print("\n" + pd.DataFrame(summaries)[cols].round(3).to_string(index=False))
    print(f"\nDetalle en {out_dir}  |  Historial en {it}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="nombre de la iteración, p. ej. it3_vision_v2")
    p.add_argument("--methods", nargs="+", default=["baseline", "ocr_llm", "vision"],
                   choices=["baseline", "ocr_llm", "vision"])
    p.add_argument("--prompt", default="v2_fel_rules")
    p.add_argument("--model", default=None)
    p.add_argument("--provider", default=None, choices=["gemini", "openai"])
    p.add_argument("--fewshot", type=int, default=0)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--gt", default=str(GT))
    p.add_argument("--notes", default="")
    p.add_argument("--no-cache", action="store_true")
    run(p.parse_args())
