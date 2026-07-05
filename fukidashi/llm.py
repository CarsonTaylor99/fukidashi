"""Thin Ollama client used by the context and translation passes."""

import json

import httpx

from .config import OLLAMA_KEEP_ALIVE, OLLAMA_MODEL, OLLAMA_NUM_CTX, OLLAMA_URL


class OllamaError(RuntimeError):
    pass


def chat(system: str, user: str, json_schema: dict | None = None,
         temperature: float = 0.3, timeout: float = 600.0) -> str:
    """One chat round-trip. If json_schema is given, Ollama constrains
    output to that schema (structured outputs) and the raw JSON string
    is returned."""
    body = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"temperature": temperature, "num_ctx": OLLAMA_NUM_CTX},
    }
    if json_schema is not None:
        body["format"] = json_schema
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=timeout)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise OllamaError(f"Ollama request failed: {e}") from e
    return r.json()["message"]["content"]


def chat_json(system: str, user: str, json_schema: dict, **kw) -> dict:
    raw = chat(system, user, json_schema=json_schema, **kw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise OllamaError(f"model returned invalid JSON: {raw[:200]}") from e


def available() -> bool:
    try:
        httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)
        return True
    except httpx.HTTPError:
        return False


def model_present() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)
        names = [m["name"] for m in r.json().get("models", [])]
        return any(n == OLLAMA_MODEL or n.split(":")[0] == OLLAMA_MODEL for n in names)
    except httpx.HTTPError:
        return False
