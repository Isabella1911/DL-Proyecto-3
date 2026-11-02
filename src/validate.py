"""Reglas de negocio de la FEL guatemalteca para detectar errores de extracción.

Una factura con alguna validación de severidad "error" se envía a revisión humana.
"""

import re
from datetime import date, timedelta

from . import normalize as N

IVA_RATE = 0.12
UUID_RE = re.compile(r"^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$")


def nit_check_digit(body: str) -> str:
    """Dígito verificador del NIT (módulo 11). 10 se representa como 'K'."""
    total = sum(int(d) * w for d, w in zip(body, range(len(body) + 1, 1, -1)))
    r = (11 - total % 11) % 11
    return "K" if r == 10 else str(r)


def nit_is_valid(v) -> bool:
    n = N.nit(v)
    if n is None or n == "CF":
        return n == "CF"
    if len(n) < 2 or not n[:-1].isdigit():
        return False
    return nit_check_digit(n[:-1]) == n[-1]


def dte_from_uuid(u: str):
    """En FEL la serie son los primeros 8 caracteres del número de autorización y el
    número de DTE es el valor decimal de los siguientes 8 caracteres hexadecimales."""
    u = N.uuid(u)
    if not u or not UUID_RE.match(u):
        return None, None
    return u[:8], str(int(u[9:13] + u[14:18], 16))


def validate(rec: dict, today: date | None = None) -> list[dict]:
    """Devuelve una lista de {campo, regla, severidad, mensaje}."""
    today = today or date.today()
    r = N.normalize_record(rec)
    out = []

    def add(field, rule, sev, msg):
        out.append({"campo": field, "regla": rule, "severidad": sev, "mensaje": msg})

    for f in ("nit_emisor", "fecha", "numero_autorizacion", "total"):
        if r[f] is None:
            add(f, "requerido", "error", "No se encontró el campo.")

    if r["nit_emisor"] is not None and not nit_is_valid(r["nit_emisor"]):
        add("nit_emisor", "nit_mod11", "error", "El dígito verificador del NIT no cuadra.")
    if r["nit_receptor"] is not None and not nit_is_valid(r["nit_receptor"]):
        add("nit_receptor", "nit_mod11", "error", "El dígito verificador del NIT no cuadra.")

    if r["numero_autorizacion"] is not None:
        if not UUID_RE.match(r["numero_autorizacion"]):
            add("numero_autorizacion", "formato_uuid", "error",
                "El número de autorización no tiene formato UUID (8-4-4-4-12).")
        else:
            serie, numero = dte_from_uuid(r["numero_autorizacion"])
            if r["serie"] is not None and r["serie"] != serie:
                add("serie", "serie_vs_autorizacion", "error",
                    f"La serie no coincide con la autorización (se esperaba {serie}).")
            if r["numero_dte"] is not None and r["numero_dte"] != numero:
                add("numero_dte", "numero_vs_autorizacion", "error",
                    f"El número de DTE no coincide con la autorización (se esperaba {numero}).")

    if r["fecha"] is not None:
        d = date.fromisoformat(r["fecha"])
        if d > today + timedelta(days=1):
            add("fecha", "fecha_futura", "error", "La fecha está en el futuro.")
        elif d < today - timedelta(days=365 * 2):
            add("fecha", "fecha_antigua", "advertencia", "La factura tiene más de 2 años.")

    if r["total"] is not None and r["total"] <= 0:
        add("total", "total_positivo", "error", "El total debe ser mayor que cero.")

    if r["iva"] is not None and r["total"] is not None and r["total"] > 0:
        esperado = r["total"] * IVA_RATE / (1 + IVA_RATE)
        if r["iva"] > r["total"]:
            add("iva", "iva_menor_total", "error", "El IVA es mayor que el total.")
        elif abs(r["iva"] - esperado) > max(0.05, 0.01 * esperado):
            # Gasolineras (IDP), exentos o pequeños contribuyentes no cumplen 12/112.
            add("iva", "iva_12_112", "advertencia",
                f"El IVA no es 12/112 del total (se esperaba ≈ Q{esperado:.2f}).")

    return out


def needs_review(issues: list[dict], strict: bool = False) -> bool:
    """strict=True también manda a revisión las advertencias."""
    sev = {"error", "advertencia"} if strict else {"error"}
    return any(i["severidad"] in sev for i in issues)


def autocorrect(rec: dict) -> dict:
    """Si la autorización es válida, de ella se derivan serie y número de DTE."""
    out = dict(rec)
    serie, numero = dte_from_uuid(rec.get("numero_autorizacion"))
    if serie:
        out["serie"] = out.get("serie") or serie
        out["numero_dte"] = out.get("numero_dte") or numero
    return out
