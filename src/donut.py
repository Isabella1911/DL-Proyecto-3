"""Modelo Donut ajustado para facturas FEL (encoder Swin + decoder BART, sin OCR).

El modelo genera una secuencia como
    <s_fel><s_nit_emisor>4809526-5</s_nit_emisor><s_nit_receptor>CF</s_nit_receptor>...</s>
que aquí se convierte en un dict con los campos de src/schema.py. Además de los valores,
devuelve la confianza de cada campo (la probabilidad mínima de sus tokens), que se usa para
marcar campos dudosos y mandarlos a revisión humana.

Con decodificación restringida (DONUT_CONSTRAINED=1), mientras el modelo escribe
un campo solo puede elegir tokens que mantengan el formato válido de ese campo: el número de
autorización solo con hexadecimal en grupos 8-4-4-4-12, la fecha AAAA-MM-DD, los montos con
dos decimales, etc. Así una "I" leída en lugar de "1" se convierte en el carácter válido más
probable, y nunca sale un valor con formato imposible.

Entrenamiento: train/train_donut.py.
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path

from . import config
from .schema import FIELDS

TASK_TOKEN = "<s_fel>"
HEX = "0123456789ABCDEF"
FIELD_TOKENS = [t for f in FIELDS for t in (f"<s_{f}>", f"</s_{f}>")]
MAX_LENGTH = 160


def to_sequence(gt: dict) -> str:
    """{campo: valor} -> '<s_campo>valor</s_campo>...'. Un valor nulo queda vacío."""
    return "".join(f"<s_{f}>{'' if gt.get(f) is None else gt[f]}</s_{f}>" for f in FIELDS)


def from_sequence(seq: str) -> dict:
    out = {}
    for f in FIELDS:
        m = re.search(rf"<s_{f}>(.*?)</s_{f}>", seq, flags=re.S)
        val = m.group(1).strip() if m else ""
        out[f] = val or None
    return out


# Formato de cada campo: (prefijo válido, valor completo). nombre_emisor no se restringe.
# Las plantillas usan H = hexadecimal y D = dígito.
FORMATS = {
    "numero_autorizacion": ("tmpl", "HHHHHHHH-HHHH-HHHH-HHHH-HHHHHHHHHHHH"),
    "serie": ("tmpl", "HHHHHHHH"),
    "fecha": ("tmpl", "DDDD-DD-DD"),
    "numero_dte": ("re", r"\d{0,10}", r"\d{1,10}"),
    "total": ("re", r"\d{0,7}(\.\d{0,2})?", r"\d{1,7}\.\d{2}"),
    "iva": ("re", r"\d{0,7}(\.\d{0,2})?", r"(\d{1,7}\.\d{2})?"),
    "nit_emisor": ("re", r"\d{0,12}(-[\dK]?)?", r"\d{1,12}-[\dK]"),
    "nit_receptor": ("re", r"\d{0,12}(-[\dK]?)?|C|CF", r"\d{1,12}-[\dK]|CF"),
}
_CHARSET = {"numero_autorizacion": HEX + "-", "serie": HEX, "fecha": "0123456789-",
            "numero_dte": "0123456789", "total": "0123456789.", "iva": "0123456789.",
            "nit_emisor": "0123456789-K", "nit_receptor": "0123456789-KCF"}


def _tmpl_ok(t: str, s: str) -> bool:
    if len(s) > len(t):
        return False
    for c, k in zip(s, t):
        if (k == "H" and c not in HEX) or (k == "D" and not c.isdigit()) or (k not in "HD" and c != k):
            return False
    return True


def prefix_ok(field: str, s: str) -> bool:
    f = FORMATS[field]
    return _tmpl_ok(f[1], s) if f[0] == "tmpl" else re.fullmatch(f[1], s) is not None


def complete_ok(field: str, s: str) -> bool:
    f = FORMATS[field]
    return (len(s) == len(f[1]) and _tmpl_ok(f[1], s)) if f[0] == "tmpl" else re.fullmatch(f[2], s) is not None


class FieldFormatProcessor:
    """LogitsProcessor que solo deja pasar tokens que mantienen el formato del campo actual."""

    def __init__(self, tok, open_ids: dict, close_ids: dict):
        self.open_ids, self.close_by_field = open_ids, close_ids
        self.close_set = set(close_ids.values())
        pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
        self.text = [p.replace("\u2581", " ") if isinstance(p, str) else "" for p in pieces]
        # Candidatos por campo: tokens hechos solo de caracteres permitidos (con espacio inicial opcional)
        self.cands = {}
        for field, chars in _CHARSET.items():
            ok = set(chars)
            self.cands[field] = [i for i, t in enumerate(self.text)
                                 if t.lstrip(" ") and len(t) - len(t.lstrip(" ")) <= 1 and set(t.lstrip(" ")) <= ok]

    def _state(self, ids):
        for k in range(len(ids) - 1, -1, -1):
            tid = ids[k]
            if tid in self.close_set:
                return None, None
            if tid in self.open_ids:
                field = self.open_ids[tid]
                return field, "".join(self.text[t] for t in ids[k + 1:]).lstrip(" ")
        return None, None

    def __call__(self, input_ids, scores):
        import torch

        for b, row in enumerate(input_ids.tolist()):
            field, cur = self._state(row)
            if field not in FORMATS:
                continue
            allowed = []
            for tid in self.cands[field]:
                t = self.text[tid]
                if t.startswith(" "):
                    if cur:
                        continue
                    t = t[1:]
                if prefix_ok(field, cur + t):
                    allowed.append(tid)
            if complete_ok(field, cur):
                allowed.append(self.close_by_field[field])
            if not allowed:  # no debería pasar; mejor no restringir que bloquear la salida
                continue
            idx = torch.tensor(allowed, device=scores.device)
            keep = scores[b, idx].clone()
            scores[b] = float("-inf")
            scores[b, idx] = keep
        return scores


@dataclass
class DonutOutput:
    fields: dict
    confidence: dict  # campo -> probabilidad mínima de sus tokens (0 a 1)
    sequence: str
    latency_s: float


class DonutFEL:
    def __init__(self, model_dir: str | Path | None = None, device: str | None = None,
                 constrained: bool | None = None):
        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        self.torch = torch
        model_dir = Path(model_dir or config.DONUT_MODEL_DIR)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"No hay un modelo Donut en {model_dir}. Entrénalo con train/train_donut.py "
                "o pon la ruta correcta en DONUT_MODEL_DIR.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = DonutProcessor.from_pretrained(model_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir, dtype=dtype).to(self.device).eval()
        tok = self.processor.tokenizer
        self.start_id = tok.convert_tokens_to_ids(TASK_TOKEN)
        self.open_ids = {tok.convert_tokens_to_ids(f"<s_{f}>"): f for f in FIELDS}
        self.close_ids = {tok.convert_tokens_to_ids(f"</s_{f}>") for f in FIELDS}
        self.name = model_dir.name
        self.constrained = config.DONUT_CONSTRAINED if constrained is None else constrained
        self.format_proc = None
        if self.constrained:
            self.format_proc = FieldFormatProcessor(
                tok, self.open_ids, {f: tok.convert_tokens_to_ids(f"</s_{f}>") for f in FIELDS})

    def predict_images(self, images: list) -> list[DonutOutput]:
        """images: lista de PIL.Image en RGB."""
        torch = self.torch
        t0 = time.perf_counter()
        pix = self.processor(images, return_tensors="pt").pixel_values.to(self.device, self.model.dtype)
        start = torch.full((len(images), 1), self.start_id, device=self.device)
        tok = self.processor.tokenizer
        extra = {}
        if getattr(self, "format_proc", None) is not None:
            from transformers import LogitsProcessorList

            extra["logits_processor"] = LogitsProcessorList([self.format_proc])
        with torch.inference_mode():
            out = self.model.generate(
                pix, decoder_input_ids=start, max_length=MAX_LENGTH, num_beams=1, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                bad_words_ids=[[tok.unk_token_id]], output_scores=True, return_dict_in_generate=True,
                **extra)
        latency = (time.perf_counter() - t0) / len(images)
        # Probabilidad del token elegido en cada paso: (batch, pasos)
        probs = torch.stack([s.float().softmax(-1) for s in out.scores], 1)
        gen = out.sequences[:, 1:]
        chosen = probs.gather(-1, gen[:, : probs.shape[1], None]).squeeze(-1).cpu().tolist()

        results = []
        for i, ids in enumerate(gen.cpu().tolist()):
            conf, field = {}, None
            for tid, p in zip(ids, chosen[i]):
                if tid in self.open_ids:
                    field = self.open_ids[tid]
                    conf[field] = p
                elif tid in self.close_ids:
                    if field:
                        conf[field] = min(conf[field], p)
                    field = None
                elif field:
                    conf[field] = min(conf[field], p)
                if tid == tok.eos_token_id:
                    break
            seq = tok.decode(ids, skip_special_tokens=False)
            seq = seq.replace(tok.eos_token, "").replace(tok.pad_token, "")
            fields = from_sequence(seq)
            results.append(DonutOutput(fields, {f: round(conf.get(f, 0.0), 4) for f in FIELDS},
                                       seq, latency))
        return results


_MODEL = None


def get_model() -> DonutFEL:
    """Carga el modelo una sola vez por proceso."""
    global _MODEL
    if _MODEL is None:
        _MODEL = DonutFEL()
    return _MODEL
