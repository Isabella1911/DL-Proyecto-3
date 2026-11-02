"""Campos que extrae el sistema y su peso de negocio."""

FIELDS = [
    "nit_emisor",
    "nit_receptor",
    "nombre_emisor",
    "fecha",
    "serie",
    "numero_dte",
    "numero_autorizacion",
    "iva",
    "total",
]

# Campos que se comparan contra el ground truth (nombre_emisor es informativo).
EVAL_FIELDS = [
    "nit_emisor",
    "nit_receptor",
    "fecha",
    "serie",
    "numero_dte",
    "numero_autorizacion",
    "iva",
    "total",
]

# Costo estimado (Q) de dejar pasar un error en cada campo sin revisión humana.
# Ajustar con la opinión de un contador; se reporta en el trabajo escrito.
ERROR_COST_Q = {
    "nit_emisor": 50.0,           # crédito fiscal mal reportado
    "nit_receptor": 20.0,
    "fecha": 15.0,                # factura en el periodo equivocado
    "serie": 5.0,
    "numero_dte": 5.0,
    "numero_autorizacion": 10.0,  # no se puede verificar en SAT
    "iva": 40.0,                  # descuadre de IVA
    "total": 40.0,                # descuadre contable
}

LABELS = {
    "nit_emisor": "NIT emisor",
    "nit_receptor": "NIT receptor",
    "nombre_emisor": "Nombre emisor",
    "fecha": "Fecha",
    "serie": "Serie",
    "numero_dte": "Número DTE",
    "numero_autorizacion": "Número de autorización",
    "iva": "IVA (Q)",
    "total": "Total (Q)",
}


def empty_result() -> dict:
    return {f: None for f in FIELDS}
