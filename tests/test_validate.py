from datetime import date

from src import normalize as N
from src.extractors import parse_json, regex_extract
from src.validate import dte_from_uuid, nit_check_digit, nit_is_valid, validate

UUID = "5C2F81A0-3E7B-4D29-A1C4-9F0E2B7D6A13"


def test_nit_mod11():
    assert nit_check_digit("576937") == "K"
    assert nit_is_valid("576937-K")
    assert nit_is_valid("c/f")
    assert not nit_is_valid("1234567-8")


def test_dte_from_uuid():
    assert dte_from_uuid(UUID.lower()) == ("5C2F81A0", str(0x3E7B4D29))


def test_normalizers():
    assert N.money("Q 1,234.50") == 1234.5
    assert N.money("1234,50") == 1234.5
    assert N.fecha("24/10/2026") == "2026-10-24"
    assert N.fecha("24-oct-2026") == "2026-10-24"
    assert N.nit("4567890 - 1") == "45678901"
    assert N.uuid(UUID.replace("-", " ").lower()) == UUID


def _good():
    serie, numero = dte_from_uuid(UUID)
    return {"nit_emisor": "576937-K", "nit_receptor": "CF", "fecha": "2026-10-24", "serie": serie,
            "numero_dte": numero, "numero_autorizacion": UUID, "iva": 10.71, "total": 100}


def test_valid_invoice_has_no_issues():
    assert validate(_good(), today=date(2026, 10, 29)) == []


def test_detects_errors():
    bad = _good() | {"nit_emisor": "576937-1", "numero_dte": "1", "iva": 30}
    rules = {i["regla"] for i in validate(bad, today=date(2026, 10, 29))}
    assert {"nit_mod11", "numero_vs_autorizacion", "iva_12_112"} <= rules


def test_parse_json_with_fences():
    assert parse_json('```json\n{"total": 5}\n```') == {"total": 5}


def test_regex_baseline():
    text = f"TIENDA\nNIT: 576937-K\nAutorizacion: {UUID}\nFecha: 24/10/2026\nNIT Cliente: C/F\nSUBTOTAL 89.29\nIVA 10.71\nTOTAL Q 100.00"
    r = regex_extract(text)
    assert r["nit_emisor"] == "576937-K" and r["nit_receptor"] == "C/F"
    assert r["numero_autorizacion"] == UUID and r["total"] == "100.00" and r["iva"] == "10.71"
