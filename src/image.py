"""Preprocesamiento de las fotos antes de mandarlas al modelo."""

import io

from PIL import Image, ImageOps

MAX_SIDE = 1600  # suficiente para leer el UUID; más grande solo gasta tokens


def load(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)  # fotos de celular vienen rotadas por EXIF
    return img.convert("RGB")


def for_llm(data: bytes, max_side: int = MAX_SIDE) -> bytes:
    img = load(data)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def for_ocr(data: bytes) -> Image.Image:
    img = load(data)
    if max(img.size) < 1800:  # Tesseract funciona mejor con texto grande
        scale = 1800 / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    gray = ImageOps.grayscale(img)
    return ImageOps.autocontrast(gray, cutoff=2)
