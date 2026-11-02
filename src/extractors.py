"""Las tres versiones del extractor que se comparan en la evaluación.

- baseline: Tesseract + expresiones regulares (sin LLM)
- ocr_llm:  Tesseract + LLM de texto que estructura el resultado
- vision:   LLM multimodal que lee la foto directamente (versión final)
"""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config, image, llm
from . import normalize as N
from .schema import empty_result
from .validate import autocorrect

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
METHODS = ["baseline", "ocr_llm", "vision"]


@dataclass
class Extraction:
    method: str
    fields: dict
    ilegibles: list = field(default_factory=list)
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model: str | None = None
    prompt: str | None = None
    raw: str | None = None
    error: str | None = None


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def parse_json(text: str) -> dict:
    """Tolera bloques ```json y texto alrededor del objeto."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _clean(data: dict) -> tuple[dict, list]:
    fields = empty_result()
    for k in fields:
        fields[k] = data.get(k)
    fields = N.normalize_record(fields)
    fields["nit_emisor"] = N.nit_display(fields["nit_emisor"])
    fields["nit_receptor"] = N.nit_display(fields["nit_receptor"])
    ileg = data.get("ilegibles") or []
    return autocorrect(fields), [i for i in ileg if isinstance(i, str)]


# --------------------------------------------------------------------------- OCR

def ocr_text(img_bytes: bytes) -> str:
    import pytesseract

    if config.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
    try:
        return pytesseract.image_to_string(image.for_ocr(img_bytes), lang="spa", config="--psm 4")
    except pytesseract.TesseractError:
        # Sin el paquete de idioma español instalado
        return pytesseract.image_to_string(image.for_ocr(img_bytes), config="--psm 4")


# ---------------------------------------------------------------------- baseline

MONEY = r"Q?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})|\d+\.\d{2})"
UUID_LOOSE = r"([0-9A-F]{8})\s*-\s*([0-9A-F]{4})\s*-\s*([0-9A-F]{4})\s*-\s*([0-9A-F]{4})\s*-\s*([0-9A-F]{12})"
NIT_RE = r"N\.?\s*I\.?\s*T\.?[^0-9A-Za-z]{0,6}(?:[A-Za-z ]{0,12}[:.]\s*)?(C\s*/?\s*F|\d{3,10}\s*-?\s*[\dKk])\b"


def regex_extract(text: str) -> dict:
    up = text.upper()
    out = empty_result()

    nits = [(m.start(), m.group(1)) for m in re.finditer(NIT_RE, up)]
    for pos, val in nits:
        ctx = up[max(0, pos - 40):pos + 5]
        is_buyer = any(w in ctx for w in ("CLIENTE", "COMPRADOR", "RECEPTOR", "NOMBRE"))
        if is_buyer or "C" in val:
            out["nit_receptor"] = out["nit_receptor"] or val
        else:
            out["nit_emisor"] = out["nit_emisor"] or val
    if out["nit_receptor"] is None and re.search(r"C\s*/\s*F|CONSUMIDOR\s+FINAL", up):
        out["nit_receptor"] = "CF"

    m = re.search(UUID_LOOSE, up)
    if m:
        out["numero_autorizacion"] = "-".join(m.groups())
    m = re.search(r"SERIE\s*[:#]?\s*([0-9A-F]{8})\b", up)
    if m:
        out["serie"] = m.group(1)
    m = re.search(r"(?:N[UÚ]MERO|NO\.)\s*(?:DE\s+)?(?:DTE|DOCUMENTO|FACTURA)?\s*[:#]?\s*(\d{6,12})\b", up)
    if m:
        out["numero_dte"] = m.group(1)

    m = re.search(r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}[-/ ][A-Z]{3}[-/ ]\d{4})", up)
    if m:
        out["fecha"] = m.group(1)

    for line in up.splitlines():
        if "TOTAL" in line and "SUB" not in line:
            vals = re.findall(MONEY, line)
            if vals:
                out["total"] = vals[-1]  # la última línea de TOTAL suele ser la final
        if re.search(r"\bI\.?V\.?A", line):
            vals = re.findall(MONEY, line)
            if vals and out["iva"] is None:
                out["iva"] = vals[-1]
    return out


def baseline(img_bytes: bytes, **_) -> Extraction:
    t0 = time.perf_counter()
    text = ocr_text(img_bytes)
    fields, _ = _clean(regex_extract(text))
    return Extraction("baseline", fields, latency_s=time.perf_counter() - t0, raw=text)


# ------------------------------------------------------------------------ LLMs

def _llm_extraction(method, parts, prompt_name, model, provider, ocr_s=0.0, raw_prefix=""):
    system = load_prompt("system")
    try:
        r = llm.generate(system, parts, provider=provider, model=model)
    except Exception as e:  # red, cuota, llave inválida
        return Extraction(method, empty_result(), prompt=prompt_name, error=f"{type(e).__name__}: {e}")
    try:
        fields, ileg = _clean(parse_json(r.text))
        err = None
    except (json.JSONDecodeError, AttributeError) as e:
        fields, ileg, err = empty_result(), [], f"JSON inválido: {e}"
    return Extraction(method, fields, ileg, r.latency_s + ocr_s, r.tokens_in, r.tokens_out,
                      r.cost_usd, r.model, prompt_name, raw_prefix + (r.text or ""), err)


def ocr_llm(img_bytes: bytes, prompt: str = "v2_fel_rules", model=None, provider=None, **_) -> Extraction:
    t0 = time.perf_counter()
    text = ocr_text(img_bytes)
    ocr_s = time.perf_counter() - t0
    user = load_prompt("ocr_wrapper").format(ocr_text=text) + "\n" + load_prompt(prompt)
    return _llm_extraction("ocr_llm", [user], prompt, model, provider, ocr_s)


def vision(img_bytes: bytes, prompt: str = "v2_fel_rules", model=None, provider=None,
           fewshot: list | None = None, **_) -> Extraction:
    """fewshot: lista de (bytes_imagen, dict_respuesta) que se muestran antes de la factura."""
    parts = []
    for i, (ex_img, ex_ans) in enumerate(fewshot or [], 1):
        parts += [f"Ejemplo {i}:", image.for_llm(ex_img),
                  "Respuesta correcta:\n" + json.dumps(ex_ans, ensure_ascii=False)]
    parts += [load_prompt(prompt), "Factura a procesar:", image.for_llm(img_bytes)]
    return _llm_extraction("vision", parts, prompt + (f"+{len(fewshot)}shot" if fewshot else ""),
                           model, provider)


EXTRACTORS = {"baseline": baseline, "ocr_llm": ocr_llm, "vision": vision}


def extract(method: str, img_bytes: bytes, **kw) -> Extraction:
    return EXTRACTORS[method](img_bytes, **kw)
