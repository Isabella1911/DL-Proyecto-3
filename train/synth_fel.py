"""Generador de facturas FEL sintéticas: imagen tipo "foto de celular" + etiquetas.

Los datos son ficticios pero consistentes con las reglas FEL que usa src/validate.py:
NIT con dígito verificador módulo 11, serie y número de DTE derivados del número de
autorización, IVA = 12/112 del total (salvo gasolineras y pequeños contribuyentes).

    python -m train.synth_fel --out data/synth --n-train 4000 --n-val 300 --workers 8

Crea data/synth/{train,val}/images/*.jpg y data/synth/{train,val}/metadata.jsonl, con una
línea por imagen: {"file": ..., "gt": {campo: valor normalizado}, "meta": {...}}.
"""

import argparse
import json
import math
import random
import sys
import textwrap
import uuid as uuidlib
from datetime import date, datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.validate import dte_from_uuid, nit_check_digit  # noqa: E402

FONTS = Path(__file__).resolve().parent / "fonts"
MONO = [("DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf"),
        ("LiberationMono-Regular.ttf", "LiberationMono-Bold.ttf"),
        ("FreeMono.otf", "FreeMonoBold.otf"),
        ("Go-Mono.ttf", "Go-Mono-Bold.ttf")]
SANS = [("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
        ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf")]

# ------------------------------------------------------------------ datos ficticios

NOMBRES = ["José", "María", "Juan", "Ana", "Carlos", "Luis", "Rosa", "Pedro", "Marta", "Jorge",
           "Sofía", "Mario", "Lucía", "Andrés", "Gabriela", "Fernando", "Carmen", "Diego", "Elena",
           "Ricardo", "Isabel", "Manuel", "Teresa", "Hugo", "Silvia", "Óscar", "Patricia", "Byron",
           "Mildred", "Edgar", "Karla", "Erick", "Wendy", "Otto", "Brenda", "Julio", "Mayra"]
APELLIDOS = ["López", "García", "Pérez", "Hernández", "Morales", "Ramírez", "Castillo", "Méndez",
             "Rodríguez", "González", "Martínez", "Cifuentes", "Barrios", "Orellana", "Monterroso",
             "Ajú", "Xicará", "Tzoc", "Coyoy", "Sajché", "Archila", "Recinos", "Estrada", "Alvarado",
             "Juárez", "Fuentes", "Chávez", "Aguilar", "Solís", "Arévalo", "Toledo", "Girón"]
APODOS = ["La Bendición", "El Sol", "San José", "Don Pedro", "La Económica", "Santa María",
          "El Buen Precio", "Los Altos", "La Esperanza", "Doña Tere", "El Progreso", "La Fe",
          "El Ahorro", "La Cosecha", "Divino Niño", "El Volcán", "La Ceiba", "El Quetzal",
          "San Antonio", "La Familia", "El Paisa", "Las Américas", "La Reforma", "Del Valle",
          "La Montaña", "El Centro", "Monte Sinaí", "La Placita", "El Rinconcito", "Mi Pueblo"]
MUNICIPIOS = ["Guatemala", "Mixco", "Villa Nueva", "San Miguel Petapa", "Quetzaltenango",
              "Antigua Guatemala", "Escuintla", "Cobán", "Huehuetenango", "Chimaltenango",
              "San Lucas Sacatepéquez", "Santa Catarina Pinula", "Fraijanes", "Retalhuleu",
              "Mazatenango", "Jalapa", "Zacapa", "Puerto Barrios", "Sololá", "Totonicapán"]
CERTIFICADORES = ["CERTIFICA DOCS, S.A.", "FACTURA DIGITAL GT, S.A.", "DOCUMENTOS EN LINEA, S.A.",
                  "SERVICIOS TRIBUTARIOS, S.A.", "E-DOCS CENTROAMERICA, S.A."]

CATEGORIAS = {
    "supermercado": (["Supermercado", "Super", "Despensa", "Abarrotería", "Tienda", "Mini Super"],
                     [("LECHE ENTERA 1L", 9, 16), ("ARROZ 1LB", 4, 7), ("FRIJOL NEGRO 1LB", 6, 10),
                      ("AZUCAR 5LB", 18, 26), ("ACEITE 900ML", 20, 32), ("HUEVOS 30U", 32, 45),
                      ("PAN SANDWICH", 14, 22), ("CAFE MOLIDO 400G", 28, 55), ("JABON BARRA", 5, 12),
                      ("PAPEL HIGIENICO 4R", 16, 30), ("POLLO ENTERO KG", 28, 40), ("TORTILLAS", 5, 10),
                      ("AGUA PURA 3L", 9, 15), ("GASEOSA 2.5L", 15, 22), ("DETERGENTE 1KG", 22, 38),
                      ("QUESO FRESCO LB", 20, 32), ("TOMATE LB", 4, 9), ("BANANO", 1, 3),
                      ("CONSOME POLLO", 8, 14), ("PASTA 200G", 4, 8), ("CREMA 250ML", 9, 15)]),
    "farmacia": (["Farmacia", "Farmacias", "Droguería"],
                 [("ACETAMINOFEN 500MG", 12, 35), ("IBUPROFENO 400MG", 18, 45), ("SUERO ORAL", 8, 16),
                  ("VITAMINA C", 25, 80), ("ALCOHOL 250ML", 10, 18), ("ANTIGRIPAL", 15, 40),
                  ("CURITAS", 8, 20), ("JARABE TOS", 35, 90), ("OMEPRAZOL 20MG", 20, 60),
                  ("PROTECTOR SOLAR", 60, 140), ("GEL ANTIBACTERIAL", 15, 30)]),
    "restaurante": (["Restaurante", "Comedor", "Cafetería", "Pollo", "Pizzería", "Taquería", "Café"],
                    [("DESAYUNO CHAPIN", 30, 55), ("CAFE AMERICANO", 12, 25), ("PEPIAN", 45, 80),
                     ("HAMBURGUESA", 40, 75), ("PIZZA MEDIANA", 70, 140), ("REFRESCO NATURAL", 10, 22),
                     ("POLLO FRITO 2PZ", 30, 50), ("TACOS (3)", 25, 45), ("CHURRASCO", 70, 120),
                     ("LICUADO", 18, 30), ("PAPAS FRITAS", 15, 28), ("POSTRE", 18, 40),
                     ("PROPINA SUGERIDA", 10, 40)]),
    "ferreteria": (["Ferretería", "Agroservicio", "Materiales", "Distribuidora"],
                   [("CEMENTO 42.5KG", 75, 95), ("CLAVO 3\" LB", 8, 14), ("PINTURA GALON", 120, 260),
                    ("TUBO PVC 1/2\"", 18, 35), ("CINTA AISLAR", 6, 15), ("FOCO LED 9W", 12, 28),
                    ("FERTILIZANTE 1QQ", 280, 420), ("MACHETE", 45, 90), ("CANDADO", 30, 80),
                    ("BROCHA 3\"", 12, 30), ("ALAMBRE AMARRE LB", 9, 16)]),
    "gasolinera": (["Gasolinera", "Estación de Servicio", "Servicentro"],
                   [("SUPER", 31, 35), ("REGULAR", 29, 33), ("DIESEL", 27, 31)]),
    "libreria": (["Librería", "Papelería", "Fotocopias"],
                 [("CUADERNO 100H", 8, 18), ("LAPICERO AZUL", 2, 5), ("RESMA PAPEL CARTA", 38, 55),
                  ("MARCADOR", 6, 14), ("FOTOCOPIAS", 1, 3), ("FOLDER", 1, 3), ("CALCULADORA", 45, 120),
                  ("ENGRAPADORA", 25, 60), ("TINTA IMPRESORA", 90, 250)]),
}
IDP_POR_GALON = {"SUPER": 4.70, "REGULAR": 4.60, "DIESEL": 1.30}

UNIDADES = ["", "UNO", "DOS", "TRES", "CUATRO", "CINCO", "SEIS", "SIETE", "OCHO", "NUEVE", "DIEZ",
            "ONCE", "DOCE", "TRECE", "CATORCE", "QUINCE", "DIECISEIS", "DIECISIETE", "DIECIOCHO",
            "DIECINUEVE", "VEINTE", "VEINTIUNO", "VEINTIDOS", "VEINTITRES", "VEINTICUATRO",
            "VEINTICINCO", "VEINTISEIS", "VEINTISIETE", "VEINTIOCHO", "VEINTINUEVE"]
DECENAS = ["", "", "", "TREINTA", "CUARENTA", "CINCUENTA", "SESENTA", "SETENTA", "OCHENTA", "NOVENTA"]
CENTENAS = ["", "CIENTO", "DOSCIENTOS", "TRESCIENTOS", "CUATROCIENTOS", "QUINIENTOS", "SEISCIENTOS",
            "SETECIENTOS", "OCHOCIENTOS", "NOVECIENTOS"]


def _letras_999(n: int) -> str:
    if n == 0:
        return ""
    if n == 100:
        return "CIEN"
    c, r = divmod(n, 100)
    if r < 30:
        resto = UNIDADES[r]
    else:
        d, u = divmod(r, 10)
        resto = DECENAS[d] + (f" Y {UNIDADES[u]}" if u else "")
    return " ".join(p for p in (CENTENAS[c], resto) if p)


def num_letras(x: float) -> str:
    """1234.5 -> 'MIL DOSCIENTOS TREINTA Y CUATRO QUETZALES CON 50/100'."""
    q, cent = int(x), round((x - int(x)) * 100)
    miles, resto = divmod(q, 1000)
    partes = []
    if miles:
        partes.append("MIL" if miles == 1 else _letras_999(miles) + " MIL")
    if resto:
        partes.append(_letras_999(resto))
    txt = " ".join(partes) or "CERO"
    return f"{txt} QUETZALES CON {cent:02d}/100"


def make_nit(rng: random.Random) -> str:
    body = str(rng.randint(100_000, 99_999_999))
    return f"{body}-{nit_check_digit(body)}"


def money(x: float, rng: random.Random, style: int) -> str:
    s = f"{x:,.2f}" if x >= 1000 and style != 2 else f"{x:.2f}"
    return ["Q", "Q ", "", "Q."][style] + s


def fmt_fecha(d: datetime, style: int) -> str:
    meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    return [d.strftime("%d/%m/%Y"), d.strftime("%d-%m-%Y"), d.strftime("%Y-%m-%d"),
            f"{d.day:02d}-{meses[d.month - 1]}-{d.year}", d.strftime("%d/%m/%Y %H:%M:%S"),
            d.strftime("%d/%m/%Y %H:%M")][style]


def make_invoice(rng: random.Random) -> dict:
    """Contenido de una factura y sus etiquetas (gt) ya normalizadas."""
    cat = rng.choice(list(CATEGORIAS))
    prefijos, productos = CATEGORIAS[cat]
    apodo = rng.choice(APODOS + [f"{rng.choice(APELLIDOS)}" for _ in range(5)])
    comercial = f"{rng.choice(prefijos)} {apodo}"
    pequeno = cat != "gasolinera" and rng.random() < 0.15
    if pequeno or rng.random() < 0.25:
        razon = f"{rng.choice(NOMBRES)} {rng.choice(APELLIDOS)} {rng.choice(APELLIDOS)}"
    else:
        razon = rng.choice([f"{rng.choice(APELLIDOS)} y {rng.choice(APELLIDOS)}, S.A.",
                            f"Inversiones {apodo}, S.A.", f"Comercializadora {rng.choice(APELLIDOS)}, S.A.",
                            f"Grupo {rng.choice(APELLIDOS)}, Sociedad Anónima"])
    show_comercial = rng.random() < 0.9

    items = []
    if cat == "gasolinera":
        prod, lo, hi = rng.choice(productos)
        precio = round(rng.uniform(lo, hi), 2)
        gal = round(rng.uniform(1.5, 15), 3)
        items.append((f"{prod} {gal:.3f} GAL", 1, precio, round(gal * precio, 2)))
        idp = round(gal * IDP_POR_GALON[prod], 2)
    else:
        idp = 0.0
        for _ in range(rng.choice([1, 1, 2, 2, 3, 3, 4, 5, 6, 8])):
            prod, lo, hi = rng.choice(productos)
            cant = rng.choice([1, 1, 1, 2, 2, 3, 4, 6])
            precio = round(rng.uniform(lo, hi), 2)
            items.append((prod, cant, precio, round(cant * precio, 2)))
    total = round(sum(i[3] for i in items), 2)
    iva = round((total - idp) * 12 / 112, 2)
    iva_desglosado = not pequeno and rng.random() < 0.75

    au = str(uuidlib.UUID(int=rng.getrandbits(128), version=4)).upper()
    serie, numero = dte_from_uuid(au)
    emision = datetime(2024, 1, 1) + timedelta(seconds=rng.randint(0, 34 * 30 * 86400))
    redondeo = rng.choice([5, 10, 50, 100])
    receptor_cf = rng.random() < 0.55
    nit_rec = "CF" if receptor_cf else make_nit(rng)
    nombre_rec = ("CONSUMIDOR FINAL" if receptor_cf
                  else f"{rng.choice(NOMBRES)} {rng.choice(APELLIDOS)}".upper())

    gt = {
        "nit_emisor": make_nit(rng),
        "nit_receptor": nit_rec,
        "nombre_emisor": comercial if show_comercial else razon,
        "fecha": emision.date().isoformat(),
        "serie": serie,
        "numero_dte": numero,
        "numero_autorizacion": au,
        "iva": f"{iva:.2f}" if iva_desglosado else None,
        "total": f"{total:.2f}",
    }
    return {
        "gt": gt, "cat": cat, "comercial": comercial, "razon": razon, "show_comercial": show_comercial,
        "pequeno": pequeno, "items": items, "total": total, "iva": iva, "idp": idp,
        "iva_desglosado": iva_desglosado, "emision": emision,
        "certificacion": emision + timedelta(seconds=rng.randint(1, 90)),
        "nombre_rec": nombre_rec,
        "direccion": f"{rng.randint(1, 30)}a. Avenida {rng.randint(1, 40)}-{rng.randint(1, 99):02d} "
                     f"Zona {rng.randint(1, 21)}, {rng.choice(MUNICIPIOS)}",
        "tel": f"{rng.randint(2200, 7999)}-{rng.randint(1000, 9999)}",
        "certificador": rng.choice(CERTIFICADORES),
        "nit_cert": make_nit(rng),
        "pago": float(math.ceil(total / redondeo) * redondeo),
    }


# ------------------------------------------------------------------- renderizado

def _qr(draw, x, y, module, rng):
    n = 25
    for i in range(n):
        for j in range(n):
            finder = (i < 7 and j < 7) or (i < 7 and j >= n - 7) or (i >= n - 7 and j < 7)
            if finder:
                a, b = i % 18 if i >= n - 7 else i, j % 18 if j >= n - 7 else j
                on = a in (0, 6) or b in (0, 6) or (2 <= a <= 4 and 2 <= b <= 4)
            else:
                on = rng.random() < 0.5
            if on:
                draw.rectangle([x + j * module, y + i * module,
                                x + (j + 1) * module - 1, y + (i + 1) * module - 1], fill=0)
    return n * module


def ticket_lines(inv: dict, rng: random.Random) -> list:
    """Líneas de un ticket térmico: (tipo, *args)."""
    m = rng.randrange(4)
    L = []
    if inv["show_comercial"]:
        L.append(("center", inv["comercial"].upper(), True, 1.3))
        L.append(("center", inv["razon"], False, 1.0))
    else:
        L.append(("center", inv["razon"].upper(), True, 1.2))
    nit_label = rng.choice(["NIT: ", "NIT ", "N.I.T. ", "NIT Emisor: "])
    L.append(("center", nit_label + inv["gt"]["nit_emisor"], False, 1.0))
    L.append(("center", inv["direccion"], False, 1.0))
    if rng.random() < 0.6:
        L.append(("center", f"Tel. {inv['tel']}", False, 1.0))
    L.append(("blank",))
    L.append(("center", rng.choice(["FACTURA", "FACTURA ELECTRONICA", "DOCUMENTO TRIBUTARIO ELECTRONICO",
                                    "FACTURA PEQUEÑO CONTRIBUYENTE" if inv["pequeno"] else "FACTURA FEL"]),
              True, 1.1))
    serie_lbl = rng.choice(["Serie: ", "SERIE: ", "Serie DTE: "])
    num_lbl = rng.choice(["Número: ", "No. DTE: ", "NUMERO: ", "No.: ", "Número DTE: "])
    aut_lbl = rng.choice(["NUMERO DE AUTORIZACION:", "No. Autorización:", "Autorización:", "UUID:",
                          "Número de Autorización:"])
    order = [("left", serie_lbl + inv["gt"]["serie"]), ("left", num_lbl + inv["gt"]["numero_dte"])]
    if rng.random() < 0.5:
        L += order
        L.append(("left", aut_lbl))
        L.append(("left", inv["gt"]["numero_autorizacion"]))
    else:
        L.append(("left", aut_lbl))
        L.append(("left", inv["gt"]["numero_autorizacion"]))
        L += order
    em_lbl = rng.choice(["Fecha: ", "Fecha emisión: ", "FECHA EMISION: ", "Fecha y hora: "])
    L.append(("left", em_lbl + fmt_fecha(inv["emision"], rng.randrange(6))))
    if rng.random() < 0.6:
        L.append(("left", "Fecha certificación: " + inv["certificacion"].strftime(
            rng.choice(["%Y-%m-%dT%H:%M:%S", "%d/%m/%Y %H:%M:%S"]))))
    L.append(("sep",))
    rec_lbl = rng.choice(["NIT: ", "NIT Cliente: ", "NIT Comprador: ", "NIT Receptor: "])
    rec_val = inv["gt"]["nit_receptor"]
    if rec_val == "CF":
        rec_val = rng.choice(["CF", "C/F", "C.F.", "CONSUMIDOR FINAL"])
    L.append(("left", rec_lbl + rec_val))
    L.append(("left", rng.choice(["Nombre: ", "Cliente: ", "NOMBRE: "]) + inv["nombre_rec"]))
    L.append(("left", "Dirección: " + rng.choice(["CIUDAD", "Ciudad", inv["direccion"].split(",")[-1].strip()])))
    L.append(("sep",))
    L.append(("lr", "CANT DESCRIPCION", "TOTAL"))
    for desc, cant, precio, sub in inv["items"]:
        if inv["cat"] == "gasolinera":
            L.append(("left", desc))
            L.append(("lr", f"   @ {money(precio, rng, m)}", money(sub, rng, m)))
        elif rng.random() < 0.5:
            L.append(("lr", f"{cant} {desc}", money(sub, rng, m)))
        else:
            L.append(("left", desc))
            L.append(("lr", f"  {cant} x {money(precio, rng, m)}", money(sub, rng, m)))
    L.append(("sep",))
    if inv["idp"]:
        L.append(("lr", "IDP:", money(inv["idp"], rng, m)))
    if inv["iva_desglosado"]:
        if rng.random() < 0.6:
            L.append(("lr", "SUBTOTAL:", money(round(inv["total"] - inv["iva"], 2), rng, m)))
        L.append(("lr", rng.choice(["IVA:", "IVA 12%:", "I.V.A.", "Total IVA:"]), money(inv["iva"], rng, m)))
    L.append(("lr", rng.choice(["TOTAL:", "TOTAL A PAGAR:", "TOTAL Q", "Total:"]),
              money(inv["total"], rng, m), True))
    if rng.random() < 0.5:
        L.append(("left", "TOTAL EN LETRAS: " + num_letras(inv["total"])))
    if rng.random() < 0.6:
        L.append(("lr", rng.choice(["EFECTIVO:", "PAGO:", "RECIBIDO:"]), money(inv["pago"], rng, m)))
        L.append(("lr", "CAMBIO:", money(round(inv["pago"] - inv["total"], 2), rng, m)))
    L.append(("blank",))
    if inv["pequeno"]:
        L.append(("center", "No genera derecho a crédito fiscal", False, 1.0))
    else:
        L.append(("center", rng.choice(["Sujeto a pagos trimestrales ISR", "Sujeto a retención definitiva ISR",
                                        "Agente de retención del IVA"]), False, 1.0))
        if not inv["iva_desglosado"]:
            L.append(("center", "Precios incluyen IVA", False, 1.0))
    L.append(("center", f"Certificador: {inv['certificador']}", False, 1.0))
    L.append(("center", f"NIT: {inv['nit_cert']}", False, 1.0))
    if rng.random() < 0.8:
        L.append(("qr",))
    L.append(("center", rng.choice(["¡Gracias por su compra!", "Vuelva pronto", "GRACIAS POR SU VISITA", ""]),
              False, 1.0))
    return L


def render_ticket(inv: dict, rng: random.Random) -> Image.Image:
    reg, bold = rng.choice(MONO)
    width = rng.choice([480, 512, 576, 576, 640])
    size = rng.randint(20, 25)
    f = {False: ImageFont.truetype(str(FONTS / reg), size), True: ImageFont.truetype(str(FONTS / bold), size)}
    margin = rng.randint(12, 28)
    cw = f[False].getlength("M")
    cols = int((width - 2 * margin) / cw)
    lh = int(size * rng.uniform(1.15, 1.4))

    lines = ticket_lines(inv, rng)
    img = Image.new("L", (width, 60 + lh * (len(lines) * 3) + 400), 255)
    d = ImageDraw.Draw(img)
    y = rng.randint(20, 50)
    for ln in lines:
        kind = ln[0]
        if kind == "blank":
            y += lh // 2
        elif kind == "sep":
            d.text((margin, y), rng.choice("-=*") * cols, font=f[False], fill=0)
            y += lh
        elif kind == "qr":
            mod = rng.randint(4, 6)
            y += 6 + _qr(d, (width - 25 * mod) // 2, y + 6, mod, rng) + 6
        elif kind == "center":
            _, text, b, scale = ln
            font = ImageFont.truetype(str(FONTS / (bold if b else reg)), int(size * scale)) if scale != 1 else f[b]
            maxc = max(8, int((width - 2 * margin) / font.getlength("M")))
            for part in textwrap.wrap(text, maxc) or [""]:
                d.text(((width - font.getlength(part)) / 2, y), part, font=font, fill=0)
                y += int(lh * scale)
        elif kind == "left":
            for part in textwrap.wrap(ln[1], cols, break_long_words=True) or [""]:
                d.text((margin, y), part, font=f[False], fill=0)
                y += lh
        elif kind == "lr":
            b = len(ln) > 3 and ln[3]
            right = ln[2]
            left = ln[1][: max(1, cols - len(right) - 1)]
            d.text((margin, y), left, font=f[b], fill=0)
            d.text((width - margin - f[b].getlength(right), y), right, font=f[b], fill=0)
            y += lh
    return img.crop((0, 0, width, y + rng.randint(20, 60)))


def render_carta(inv: dict, rng: random.Random) -> Image.Image:
    """Factura tamaño carta (PDF impreso), con recuadros y tabla."""
    reg, bold = rng.choice(SANS + MONO[:2])
    W, H = 1100, 1420
    s = rng.randint(19, 23)
    F = lambda b=False, k=1.0: ImageFont.truetype(str(FONTS / (bold if b else reg)), int(s * k))  # noqa: E731
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    g = inv["gt"]
    y = 50
    titulo = inv["comercial"] if inv["show_comercial"] else inv["razon"]
    tf = F(True, 1.4)
    while tf.getlength(titulo.upper()) > 480 and tf.size > 14:
        tf = ImageFont.truetype(tf.path, tf.size - 1)
    d.text((50, y), titulo.upper(), font=tf, fill=0)
    y += int(s * 2)
    for t in ([inv["razon"]] if inv["show_comercial"] else []) + [f"NIT: {g['nit_emisor']}", inv["direccion"],
                                                                   f"Teléfono: {inv['tel']}"]:
        d.text((50, y), t[:44], font=F(), fill=0)
        y += int(s * 1.4)
    bx = 560
    d.rectangle([bx, 40, W - 40, 300], outline=0, width=2)
    yy = 52
    for t, b in [("FACTURA ELECTRÓNICA", True), (f"Serie: {g['serie']}", False),
                 (f"Número de DTE: {g['numero_dte']}", False), ("Número de autorización:", False),
                 (g["numero_autorizacion"], False),
                 (f"Fecha de emisión: {fmt_fecha(inv['emision'], rng.randrange(6))}", False),
                 ("Fecha certificación: " + inv["certificacion"].strftime("%Y-%m-%dT%H:%M:%S"), False)]:
        font = F(b, 0.95)
        while font.getlength(t) > W - 60 - bx and font.size > 10:
            font = ImageFont.truetype(font.path, font.size - 1)
        d.text((bx + 10, yy), t, font=font, fill=0)
        yy += int(s * 1.25)
    y = max(y, 320) + 10
    d.rectangle([40, y, W - 40, y + 110], outline=0, width=2)
    rec = g["nit_receptor"] if g["nit_receptor"] != "CF" else rng.choice(["CF", "C/F"])
    d.text((55, y + 12), f"NIT receptor: {rec}", font=F(), fill=0)
    d.text((55, y + 12 + int(s * 1.4)), f"Nombre: {inv['nombre_rec']}", font=F(), fill=0)
    d.text((55, y + 12 + int(s * 2.8)), "Dirección: CIUDAD", font=F(), fill=0)
    if rng.random() < 0.4:
        venc = inv["emision"] + timedelta(days=30)
        d.text((620, y + 12), f"Vencimiento: {venc.strftime('%d/%m/%Y')}", font=F(), fill=0)
    y += 140
    cols = [55, 140, 640, 840]
    d.rectangle([40, y, W - 40, y + int(s * 1.6)], fill=225, outline=0)
    for x, t in zip(cols, ["Cant.", "Descripción", "P. unitario", "Subtotal"]):
        d.text((x, y + 5), t, font=F(True), fill=0)
    y += int(s * 1.9)
    for desc, cant, precio, sub in inv["items"]:
        for x, t in zip(cols, [str(cant), desc, f"Q {precio:,.2f}", f"Q {sub:,.2f}"]):
            d.text((x, y), t, font=F(), fill=0)
        y += int(s * 1.5)
    y += 10
    d.line([40, y, W - 40, y], fill=0, width=2)
    y += 15
    tot = []
    if inv["idp"]:
        tot.append(("IDP", inv["idp"]))
    if inv["iva_desglosado"]:
        tot += [("Subtotal", round(inv["total"] - inv["iva"], 2)), ("IVA (12%)", inv["iva"])]
    tot.append(("TOTAL", inv["total"]))
    for lbl, v in tot:
        b = lbl == "TOTAL"
        d.text((640, y), lbl, font=F(b), fill=0)
        t = f"Q {v:,.2f}"
        d.text((W - 55 - F(b).getlength(t), y), t, font=F(b), fill=0)
        y += int(s * 1.5)
    d.text((55, y + 10), "Total en letras: " + num_letras(inv["total"]).title(), font=F(k=0.85), fill=0)
    y += 70
    nota = ("Pequeño contribuyente. No genera derecho a crédito fiscal." if inv["pequeno"]
            else "Sujeto a pagos trimestrales ISR.")
    d.text((55, y), nota, font=F(k=0.85), fill=0)
    d.text((55, y + int(s * 1.3)), f"Certificador: {inv['certificador']}  NIT: {inv['nit_cert']}",
           font=F(k=0.85), fill=0)
    if rng.random() < 0.8:
        _qr(d, W - 200, y - 10, 6, rng)
    # Hoja carta completa, vertical, como se imprime y se fotografía en la realidad
    return img


# --------------------------------------------------------------- "foto" simulada

def _persp_coeffs(src, dst):
    """Coeficientes de PIL (salida -> entrada) para mapear el cuadrilátero src a dst."""
    A, B = [], []
    for (x, y), (X, Y) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
        B += [X, Y]
    return np.linalg.solve(np.array(A, float), np.array(B, float)).tolist()


def _smooth_field(rng, h, w, lo, hi, cells=4):
    """Campo aleatorio suave en [lo, hi] (para desteñido e iluminación)."""
    small = np.array([[rng.uniform(lo, hi) for _ in range(cells)] for _ in range(cells)], np.float32)
    return np.asarray(Image.fromarray(small).resize((w, h), Image.BICUBIC), np.float32).clip(min(lo, hi), max(lo, hi))


def _background(rng, w, h):
    base = np.array(rng.choice([(120, 85, 55), (160, 120, 80), (90, 90, 95), (200, 200, 195),
                                (40, 40, 45), (70, 100, 140), (150, 40, 40), (230, 225, 210)]), np.float32)
    base += np.array([rng.uniform(-20, 20) for _ in range(3)], np.float32)
    bg = np.ones((h, w, 3), np.float32) * base
    bg *= _smooth_field(rng, h, w, 0.75, 1.15, 3)[..., None]
    bg += np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(2, 10), (h, w, 1))
    if rng.random() < 0.4:  # vetas de madera / tela
        yy = np.arange(h, dtype=np.float32)[:, None]
        bg *= (1 + 0.06 * np.sin(yy / rng.uniform(4, 25) + rng.uniform(0, 6)))[..., None]
    return bg


def photograph(doc: Image.Image, rng: random.Random, max_side: int = 1600) -> Image.Image:
    nrng = np.random.default_rng(rng.randrange(2**32))
    w, h = doc.size
    ink = 1 - np.asarray(doc, np.float32) / 255
    ink *= _smooth_field(rng, h, w, rng.uniform(0.3, 0.8), 1.0, 5) if rng.random() < 0.5 else rng.uniform(0.75, 1)
    paper = np.array([rng.uniform(235, 255), rng.uniform(232, 252), rng.uniform(220, 248)], np.float32)
    rgb = paper * (1 - ink[..., None] * rng.uniform(0.8, 0.95))
    if rng.random() < 0.4:  # arrugas / dobleces
        for _ in range(rng.randint(1, 3)):
            yy = rng.uniform(0, h)
            band = np.exp(-((np.arange(h) - yy) ** 2) / (2 * rng.uniform(3, 15) ** 2)).astype(np.float32)
            rgb *= (1 - rng.uniform(0.05, 0.15) * band)[:, None, None]
    doc_rgba = Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).convert("RGBA")

    if rng.random() < 0.15:  # recorte tipo "escáner" / WhatsApp recortado
        angle = rng.uniform(-2, 2)
        out = doc_rgba.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(245, 245, 240, 255))
        photo = np.asarray(out.convert("RGB"), np.float32)
    else:
        fill = rng.uniform(0.7, 0.97)
        # ancho / alto de la foto: retrato para tickets, más ancho para facturas carta
        aspect = min(1.34, max(0.5625, w / h * rng.uniform(1.0, 1.2)))
        if h / w > 1 / aspect:
            ph = int(h / fill)
            pw = int(ph * aspect)
        else:
            pw = int(w / fill)
            ph = int(pw / aspect)
        pw, ph = max(pw, w + 20), max(ph, h + 20)
        bg = _background(rng, pw, ph)
        cx, cy = pw / 2 + rng.uniform(-0.05, 0.05) * pw, ph / 2 + rng.uniform(-0.03, 0.03) * ph
        a = math.radians(rng.uniform(-7, 7))
        corners = []
        for x, y in [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]:
            jx, jy = rng.uniform(-0.04, 0.04) * w, rng.uniform(-0.02, 0.02) * h
            corners.append((cx + (x + jx) * math.cos(a) - (y + jy) * math.sin(a),
                            cy + (x + jx) * math.sin(a) + (y + jy) * math.cos(a)))
        warped = doc_rgba.transform((pw, ph), Image.PERSPECTIVE,
                                    _persp_coeffs([(0, 0), (w, 0), (w, h), (0, h)], corners), Image.BICUBIC)
        alpha = np.asarray(warped, np.float32)[..., 3:] / 255
        photo = bg * (1 - alpha) + np.asarray(warped, np.float32)[..., :3] * alpha
        if rng.random() < 0.35:  # sombra del teléfono / la mano
            mask = Image.new("L", (pw, ph), 0)
            x0 = rng.uniform(-0.2, 0.6) * pw
            ImageDraw.Draw(mask).polygon([(x0, -10), (x0 + rng.uniform(0.3, 0.8) * pw, -10),
                                          (x0 + rng.uniform(0.1, 0.9) * pw, ph * rng.uniform(0.3, 1.1)),
                                          (x0 - rng.uniform(0, 0.3) * pw, ph * rng.uniform(0.3, 1.1))], fill=255)
            m = np.asarray(mask.filter(ImageFilter.GaussianBlur(rng.uniform(15, 60))), np.float32) / 255
            photo *= (1 - rng.uniform(0.2, 0.45) * m)[..., None]

    ph, pw = photo.shape[:2]
    photo *= _smooth_field(rng, ph, pw, rng.uniform(0.7, 0.95), rng.uniform(1.0, 1.15), 3)[..., None]
    photo *= np.array([rng.uniform(0.92, 1.08) for _ in range(3)], np.float32)  # balance de blancos
    img = Image.fromarray(photo.clip(0, 255).astype(np.uint8))
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    r = rng.choice([0, 0, 0.4, 0.7, 1.0, 1.4])
    if r:
        img = img.filter(ImageFilter.GaussianBlur(r))
    arr = np.asarray(img, np.float32) + nrng.normal(0, rng.uniform(1, 7), (img.height, img.width, 1))
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------- dataset

def make_sample(args):
    idx, seed, out_dir, carta_frac = args
    rng = random.Random(seed)
    inv = make_invoice(rng)
    layout = "carta" if rng.random() < carta_frac else "ticket"
    doc = render_carta(inv, rng) if layout == "carta" else render_ticket(inv, rng)
    img = photograph(doc, rng)
    name = f"synth_{idx:05d}.jpg"
    img.save(out_dir / "images" / name, quality=rng.randint(45, 92))
    return {"file": f"images/{name}", "gt": inv["gt"],
            "meta": {"layout": layout, "categoria": inv["cat"], "seed": seed}}


def build(out: Path, split: str, n: int, base_seed: int, workers: int, carta_frac: float = 0.2):
    d = out / split
    (d / "images").mkdir(parents=True, exist_ok=True)
    jobs = [(i, base_seed * 1_000_003 + i, d, carta_frac) for i in range(n)]
    with Pool(workers) as pool, (d / "metadata.jsonl").open("w", encoding="utf-8") as fh:
        for k, rec in enumerate(pool.imap(make_sample, jobs, chunksize=8), 1):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k % 250 == 0 or k == n:
                print(f"  {split}: {k}/{n}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "data" / "synth"))
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-val", type=int, default=300)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--carta-frac", type=float, default=0.2, help="proporción de facturas tamaño carta")
    a = p.parse_args()
    out = Path(a.out)
    build(out, "train", a.n_train, a.seed, a.workers, a.carta_frac)
    # Validación siempre con 20 % carta (la proporción "real" supuesta), para comparar iteraciones
    build(out, "val", a.n_val, a.seed + 1, a.workers)  # semilla distinta: sin facturas repetidas
    print(f"Listo: {out}")
