"""Cliente mínimo para LLMs con visión (Gemini u OpenAI)."""

import base64
import time

from . import config


class LLMResponse:
    def __init__(self, text, tokens_in, tokens_out, latency_s, model):
        self.text = text
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.latency_s = latency_s
        self.model = model

    @property
    def cost_usd(self):
        return config.cost_usd(self.model, self.tokens_in, self.tokens_out)


def generate(system: str, parts: list, temperature: float = 0.0, max_tokens: int = 1024,
             provider: str | None = None, model: str | None = None) -> LLMResponse:
    """parts: lista de str (texto) o bytes (imagen JPEG), en orden."""
    provider = provider or config.PROVIDER
    if provider == "gemini":
        return _gemini(system, parts, temperature, max_tokens, model or config.GEMINI_MODEL)
    if provider == "openai":
        return _openai(system, parts, temperature, max_tokens, model or config.OPENAI_MODEL)
    raise ValueError(f"Proveedor desconocido: {provider}")


def _gemini(system, parts, temperature, max_tokens, model):
    from google import genai
    from google.genai import types

    client = genai.Client()  # lee GEMINI_API_KEY / GOOGLE_API_KEY
    contents = [
        types.Part.from_bytes(data=p, mime_type="image/jpeg") if isinstance(p, bytes) else p
        for p in parts
    ]
    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            # Sin "razonamiento" extra: la tarea es de lectura y baja la latencia.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    latency = time.perf_counter() - t0
    u = resp.usage_metadata
    return LLMResponse(resp.text, u.prompt_token_count or 0,
                       (u.candidates_token_count or 0) + (u.thoughts_token_count or 0),
                       latency, model)


def _openai(system, parts, temperature, max_tokens, model):
    from openai import OpenAI

    client = OpenAI()  # lee OPENAI_API_KEY
    content = []
    for p in parts:
        if isinstance(p, bytes):
            b64 = base64.b64encode(p).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}})
        else:
            content.append({"type": "text", "text": p})
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    latency = time.perf_counter() - t0
    return LLMResponse(resp.choices[0].message.content, resp.usage.prompt_tokens,
                       resp.usage.completion_tokens, latency, model)
