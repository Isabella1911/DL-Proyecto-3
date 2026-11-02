"""App web: sube fotos de facturas FEL y obtén sus campos listos para Excel.

Ejecutar:  streamlit run app/app.py
"""

import hashlib
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.extractors import Extraction, extract  # noqa: E402
from src.schema import FIELDS, LABELS  # noqa: E402
from src.validate import needs_review, validate  # noqa: E402

CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)
MAX_MB = 15

METHOD_LABELS = {
    "donut": "Modelo propio Donut (local, sin internet)",
    "vision": "LLM de visión (recomendado)",
    "ocr_llm": "OCR + LLM de texto",
    "baseline": "OCR + reglas (sin LLM)",
}

st.set_page_config(page_title="FacturaFEL", page_icon="🧾", layout="wide")


# ----------------------------------------------------------------- extracción

def cache_key(data: bytes, method: str, prompt: str, model: str) -> Path:
    h = hashlib.sha1(data + f"|{method}|{prompt}|{model}".encode()).hexdigest()[:20]
    return CACHE / f"{h}.json"


def run_extraction(data: bytes, method: str, prompt: str, offline: bool) -> Extraction:
    model = {"baseline": "-", "donut": config.DONUT_MODEL_DIR}.get(method, config.model_name())
    key = cache_key(data, method, prompt, model)
    if key.exists():
        return Extraction(**json.loads(key.read_text(encoding="utf-8")))
    if offline and method != "donut":  # Donut corre local: no necesita internet
        return Extraction(method, {f: None for f in FIELDS},
                          error="Modo sin internet: esta foto no está en caché.")
    ex = extract(method, data, prompt=prompt)
    if not ex.error:
        key.write_text(json.dumps(asdict(ex), ensure_ascii=False), encoding="utf-8")
    return ex


def friendly_error(err: str) -> str:
    e = err.lower()
    if "api key" in e or "api_key" in e or "401" in e or "permission" in e:
        return "No hay una llave de API válida. Revisa el archivo .env (GEMINI_API_KEY u OPENAI_API_KEY)."
    if "429" in e or "quota" in e or "rate" in e:
        return "Se alcanzó el límite de solicitudes de la API. Espera unos segundos e intenta de nuevo."
    if "tesseract" in e:
        return "Tesseract no está instalado o no se encontró. Usa el método 'LLM de visión'."
    if "no hay un modelo donut" in e:
        return err
    if "json" in e:
        return "El modelo devolvió una respuesta con formato inválido. Intenta de nuevo."
    return err


# ----------------------------------------------------------------------- UI

with st.sidebar:
    st.header("Configuración")
    method = st.radio("Método", list(METHOD_LABELS), format_func=METHOD_LABELS.get)
    prompt = st.selectbox("Versión del prompt", ["v2_fel_rules", "v1_zero_shot"],
                          disabled=method in ("baseline", "donut"))
    offline = st.toggle("Modo sin internet (usar caché)", value=False,
                        help="Para la feria: responde solo con fotos procesadas antes.")
    st.caption(f"Modelo: `{config.model_name()}` · proveedor `{config.PROVIDER}`")
    if st.button("Limpiar tabla"):
        st.session_state.pop("rows", None)

st.session_state.setdefault("rows", [])

tab_app, tab_how = st.tabs(["🧾 Procesar facturas", "📊 Cómo funciona y métricas"])

with tab_app:
    st.title("FacturaFEL")
    st.write("Sube la foto de una factura electrónica (FEL) y obtén NIT, fecha, serie, "
             "autorización, IVA y total listos para tu hoja de cálculo.")

    c1, c2 = st.columns(2)
    files = c1.file_uploader("Fotos de facturas", type=["jpg", "jpeg", "png", "webp"],
                             accept_multiple_files=True)
    cam = c2.camera_input("O toma una foto", label_visibility="collapsed") if c2.toggle("Usar cámara") else None
    uploads = list(files or []) + ([cam] if cam else [])

    for up in uploads:
        data = up.getvalue()
        name = getattr(up, "name", "camara.jpg")
        st.divider()
        if len(data) > MAX_MB * 1024 * 1024:
            st.error(f"{name}: la imagen pesa más de {MAX_MB} MB.")
            continue
        left, right = st.columns([1, 2])
        try:
            left.image(data, caption=name, width="stretch")
        except Exception:
            st.error(f"{name}: no se pudo abrir como imagen.")
            continue

        with right, st.spinner("Leyendo la factura..."):
            try:
                ex = run_extraction(data, method, prompt, offline)
            except Exception as e:  # p. ej. Tesseract no instalado
                ex = Extraction(method, {f: None for f in FIELDS}, error=f"{type(e).__name__}: {e}")

        if ex.error:
            right.error(friendly_error(ex.error))
            continue

        issues = validate(ex.fields)
        bad = {i["campo"] for i in issues} | set(ex.ilegibles)
        if needs_review(issues):
            right.warning("⚠️ **Revisar antes de registrar.** "
                          + " ".join(i["mensaje"] for i in issues if i["severidad"] == "error"))
        elif bad:
            right.info("ℹ️ Lista, con observaciones: "
                       + " ".join(i["mensaje"] for i in issues) + (f" Campos dudosos: {', '.join(ex.ilegibles)}." if ex.ilegibles else ""))
        else:
            right.success("✅ Todas las validaciones pasaron. Lista para contabilidad.")

        table = pd.DataFrame({
            "Campo": [LABELS[f] for f in FIELDS],
            "Valor": [None if ex.fields.get(f) is None else str(ex.fields.get(f)) for f in FIELDS],
            "Estado": ["⚠️ revisar" if f in bad else ("—" if ex.fields.get(f) is None else "✓") for f in FIELDS],
        })
        edited = right.data_editor(table, hide_index=True, disabled=["Campo", "Estado"],
                                   key=f"ed_{hashlib.sha1(data).hexdigest()[:10]}",
                                   width="stretch")
        right.caption(f"⏱ {ex.latency_s:.1f} s · {ex.tokens_in + ex.tokens_out:,} tokens · "
                      f"Q{ex.cost_usd * config.USD_TO_GTQ:.4f} · {METHOD_LABELS[method]}")

        if right.button("Agregar a la tabla", key=f"add_{name}_{len(data)}"):
            row = {"archivo": name, **dict(zip(FIELDS, edited["Valor"])),
                   "requiere_revision": needs_review(issues)}
            st.session_state.rows.append(row)
            st.toast("Agregada")

    if st.session_state.rows:
        st.divider()
        st.subheader(f"Facturas capturadas ({len(st.session_state.rows)})")
        df = pd.DataFrame(st.session_state.rows)
        st.dataframe(df, hide_index=True, width="stretch")
        xls = io.BytesIO()
        df.to_excel(xls, index=False)
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Descargar Excel", xls.getvalue(), "facturas_fel.xlsx")
        d2.download_button("⬇️ Descargar CSV", df.to_csv(index=False).encode("utf-8-sig"), "facturas_fel.csv")

    st.caption("Las fotos no se guardan en ningún servidor propio; se envían al proveedor del LLM "
               "solo para procesarlas. Revisa siempre los campos marcados antes de registrarlos.")

with tab_how:
    st.subheader("Pipeline")
    st.graphviz_chart("""
digraph {
  rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica", fontsize=16];
  foto [label="Foto de la factura\\n(celular / WhatsApp)"];
  pre [label="Preprocesamiento\\nrotación EXIF, resize 1600px, JPEG"];
  llm [label="LLM multimodal\\nprompt de sistema + reglas FEL\\nsalida JSON, temperature=0"];
  norm [label="Normalización\\nNIT, fecha ISO, montos, UUID"];
  val [label="Validaciones FEL\\nNIT módulo 11 · UUID\\nserie/número vs autorización\\nIVA = 12/112 del total"];
  ok [label="Lista para contabilidad", fillcolor="#dcfce7"];
  rev [label="Revisión humana", fillcolor="#fef3c7"];
  xls [label="Excel / CSV", fillcolor="#e0f2fe"];
  foto -> pre -> llm -> norm -> val;
  val -> ok [label="pasa"]; val -> rev [label="falla"];
  ok -> xls; rev -> xls;
}""")

    it = ROOT / "results" / "iterations.csv"
    if it.exists():
        res = pd.read_csv(it)
        st.subheader("Resultados por iteración")
        show = ["iteracion", "metodo", "modelo", "prompt", "n", "f1", "exactitud_documento",
                "acc_nit_emisor", "acc_total", "latencia_media_s", "costo_usd_por_factura",
                "pct_revision_errores", "ahorro_Q_1000_errores"]
        st.dataframe(res[[c for c in show if c in res]].round(3), hide_index=True, width="stretch")
        res["corrida"] = res["iteracion"] + " · " + res["metodo"]
        m1, m2 = st.columns(2)
        m1.caption("F1 por campo (micro)")
        m1.bar_chart(res.set_index("corrida")["f1"])
        m2.caption("Ahorro neto por 1,000 facturas (Q), revisando las que fallan validación")
        m2.bar_chart(res.set_index("corrida")["ahorro_Q_1000_errores"])
        acc = res.set_index("corrida")[[c for c in res if c.startswith("acc_")]].T
        st.caption("Exactitud por campo")
        st.dataframe(acc.round(3), width="stretch")
    else:
        st.info("Todavía no hay resultados. Corre `python -m eval.run_eval --tag it1 ...`.")
