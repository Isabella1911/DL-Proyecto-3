"""Normaliza los valores extraídos para que se puedan comparar y exportar."""

import re
from datetime import date

MESES = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "set": 9, "oct": 10, "nov": 11, "dic": 12,
}


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and v != v) or str(v).strip() in ("", "null", "None", "nan")


def nit(v):
    """'1234567-8' / '1234567 8' / 'c/f' -> '12345678' o 'CF'."""
    if _blank(v):
        return None
    s = re.sub(r"[^0-9Kk/CcFf]", "", str(v)).upper().replace("/", "")
    if s in ("CF", "C", "CONSUMIDORFINAL") or "CONSUMIDOR" in str(v).upper():
        return "CF"
    s = re.sub(r"[^0-9K]", "", s)
    return s.lstrip("0") or None


def nit_display(v):
    n = nit(v)
    if n is None or n == "CF":
        return n
    return f"{n[:-1]}-{n[-1]}" if len(n) > 1 else n


def fecha(v):
    """Acepta dd/mm/aaaa, dd-mm-aaaa, aaaa-mm-dd y '20-oct-2026'. Devuelve 'aaaa-mm-dd'."""
    if _blank(v):
        return None
    s = str(v).strip().lower()
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = re.search(r"(\d{1,2})[-/. ](\d{1,2}|[a-zá-ú]{3,})[-/. ,]+(\d{2,4})", s)
        if not m:
            return None
        d, mo_s, y = int(m.group(1)), m.group(2), int(m.group(3))
        mo = int(mo_s) if mo_s.isdigit() else MESES.get(mo_s[:3])
        if mo is None:
            return None
        if y < 100:
            y += 2000
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return None


def money(v):
    """'Q 1,234.50' -> 1234.5"""
    if _blank(v):
        return None
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    s = re.sub(r"[^0-9.,-]", "", str(v))
    if "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s:
        # '1234,50' (coma decimal) vs '1,234' (miles)
        s = s.replace(",", ".") if re.search(r",\d{2}$", s) else s.replace(",", "")
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def uuid(v):
    if _blank(v):
        return None
    s = re.sub(r"[^0-9A-Fa-f]", "", str(v)).upper()
    if len(s) != 32:
        return str(v).strip().upper()
    return f"{s[:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:]}"


def serie(v):
    if _blank(v):
        return None
    return re.sub(r"\s", "", str(v)).upper()


def numero_dte(v):
    if _blank(v):
        return None
    s = re.sub(r"\D", "", str(v).split(".")[0] if isinstance(v, float) else str(v))
    return s.lstrip("0") or None


def text(v):
    if _blank(v):
        return None
    return " ".join(str(v).split())


NORMALIZERS = {
    "nit_emisor": nit,
    "nit_receptor": nit,
    "nombre_emisor": text,
    "fecha": fecha,
    "serie": serie,
    "numero_dte": numero_dte,
    "numero_autorizacion": uuid,
    "iva": money,
    "total": money,
}


def normalize_record(rec: dict) -> dict:
    return {k: NORMALIZERS.get(k, text)(rec.get(k)) for k in NORMALIZERS}


def equal(field: str, pred, gold) -> bool:
    p = NORMALIZERS[field](pred)
    g = NORMALIZERS[field](gold)
    if field in ("iva", "total") and p is not None and g is not None:
        return abs(p - g) <= 0.01
    return p == g
