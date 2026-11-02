"""Configuración leída de variables de entorno (archivo .env)."""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # gemini | openai
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TESSERACT_CMD = os.getenv("TESSERACT_CMD")  # en Windows: C:\Program Files\Tesseract-OCR\tesseract.exe

# Precio en USD por millón de tokens (entrada, salida). Verificar en la página de
# precios del proveedor antes de reportar costos.
PRICES_USD_PER_M = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}
USD_TO_GTQ = float(os.getenv("USD_TO_GTQ", "7.7"))

# Modelo Donut propio (train/train_donut.py). Un campo con confianza menor al umbral se
# marca como dudoso y la factura va a revisión con la política "estricta".
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONUT_MODEL_DIR = os.path.join(_ROOT, os.getenv("DONUT_MODEL_DIR", os.path.join("models", "donut-fel")))
DONUT_MIN_CONF = float(os.getenv("DONUT_MIN_CONF", "0.9"))  # óptimo según docs/donut/curva_valor_it2.png
# Decodificación restringida al formato de cada campo (ver src/donut.py). Apagada por defecto:
# ver la comparación de valor en RESULTADOS.md.
DONUT_CONSTRAINED = os.getenv("DONUT_CONSTRAINED", "0") == "1"

# Modelo de valor: costo de digitar una factura a mano y de revisarla.
MANUAL_MINUTES_PER_INVOICE = float(os.getenv("MANUAL_MINUTES_PER_INVOICE", "2.0"))
REVIEW_MINUTES_PER_INVOICE = float(os.getenv("REVIEW_MINUTES_PER_INVOICE", "0.75"))
HOURLY_WAGE_GTQ = float(os.getenv("HOURLY_WAGE_GTQ", "30"))


def model_name() -> str:
    return GEMINI_MODEL if PROVIDER == "gemini" else OPENAI_MODEL


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = PRICES_USD_PER_M.get(model, (0.0, 0.0))
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000
