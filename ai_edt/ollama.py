"""Ollama API client for AI-EDT.

Single call point for all LLM inference. Keeps VRAM clean by defaulting
keep_alive=0 — models are unloaded immediately after each call, preventing
collisions between the 1B sieve and 8B reasoning engine on 4 GB VRAM.
"""

from __future__ import annotations

import requests

from ai_edt.config import get_config
from ai_edt.logger import get_logger

logger = get_logger("ollama")


class OllamaError(Exception):
    """Raised when the Ollama API call fails."""


def generate(
    prompt: str,
    model: str,
    timeout: int,
    keep_alive: int = 0,
    temperature: float | None = None,
) -> str:
    """Call Ollama /api/generate and return the response text.

    Args:
        prompt:      The prompt string to send.
        model:       Ollama model tag (e.g. "llama3.2:1b").
        timeout:     Request timeout in seconds.
        keep_alive:  Seconds to keep model in VRAM after inference.
                     0 = unload immediately (default, required for 4 GB VRAM).
        temperature: Sampling temperature. None uses the model's default.

    Returns:
        The model's response text (DeepSeek <think> blocks are NOT stripped
        here — that is the caller's responsibility).

    Raises:
        OllamaError: On connection failure, timeout, or malformed response.
    """
    cfg = get_config()
    url = f"{cfg.ollama_host}/api/generate"

    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
    }
    if temperature is not None:
        payload["options"] = {"temperature": temperature}

    logger.debug("Ollama call | model=%s | timeout=%ds", model, timeout)

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        raise OllamaError(f"Timed out after {timeout}s (model={model})") from None
    except requests.exceptions.ConnectionError:
        raise OllamaError(
            f"Cannot connect to Ollama at {cfg.ollama_host}. Is `ollama serve` running?"
        ) from None
    except (requests.exceptions.RequestException, ValueError) as exc:
        raise OllamaError(f"Request failed: {exc}") from exc

    text = data.get("response", "")
    if not text:
        raise OllamaError(f"Empty response from Ollama (model={model})")

    return text
