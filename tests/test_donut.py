import random
from datetime import date

from src.donut import from_sequence, to_sequence
from src.schema import FIELDS
from src.validate import validate
from train.synth_fel import make_invoice, num_letras


def test_sequence_roundtrip():
    gt = {"nit_emisor": "576937-K", "nit_receptor": "CF", "nombre_emisor": "Super El Sol",
          "fecha": "2026-10-24", "serie": "5C2F81A0", "numero_dte": "1048268073",
          "numero_autorizacion": "5C2F81A0-3E7B-4D29-A1C4-9F0E2B7D6A13", "iva": None, "total": "100.00"}
    seq = to_sequence(gt)
    assert seq.startswith("<s_nit_emisor>576937-K</s_nit_emisor>")
    assert "<s_iva></s_iva>" in seq
    assert from_sequence(seq) == gt


def test_sequence_tolerates_missing_and_truncated_fields():
    out = from_sequence("<s_nit_emisor>123-4</s_nit_emisor><s_total>5.00</s_total><s_fecha>2026")
    assert out["nit_emisor"] == "123-4" and out["total"] == "5.00"
    assert out["fecha"] is None and set(out) == set(FIELDS)


def test_synthetic_invoices_pass_fel_rules():
    rng = random.Random(0)
    for _ in range(300):
        inv = make_invoice(rng)
        issues = validate(inv["gt"], today=date(2026, 12, 31))
        errores = [i for i in issues if i["severidad"] == "error"]
        assert not errores, (inv["gt"], errores)
        if inv["cat"] != "gasolinera":  # IVA = 12/112 del total salvo gasolineras (IDP)
            assert not [i for i in issues if i["regla"] == "iva_12_112"], inv["gt"]


def test_num_letras():
    assert num_letras(100) == "CIEN QUETZALES CON 00/100"
    assert num_letras(1234.5) == "MIL DOSCIENTOS TREINTA Y CUATRO QUETZALES CON 50/100"
    assert num_letras(962.19) == "NOVECIENTOS SESENTA Y DOS QUETZALES CON 19/100"
