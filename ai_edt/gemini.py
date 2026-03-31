"""Google Gemini API client for AI-EDT.

Drop-in replacement for Ollama at Stage 3. Identical call signature to
ollama.generate() so pipeline.py can switch providers with a one-line change.

Setup:
    1. pip install google-genai           (already a transitive dependency)
    2. Set GEMINI_API_KEY in your environment or a .env file.
       Get a key at: https://aistudio.google.com/apikey

Recommended model: gemini-2.0-flash
    - Fastest Gemini model, sub-3s latency for this prompt size
    - Reasoning quality far exceeds DeepSeek-R1-8B on structured financial tasks
    - Cost: ~$0.0001 per headline at Flash pricing (effectively free for testing)
"""

from __future__ import annotations

import os
import re
import threading
import time

from ai_edt.logger import get_logger

logger = get_logger("gemini")
# Semaphore: cap concurrent Gemini API calls at 3 to avoid burst rate-limit
# storms (e.g. OPEC announcement hitting all 4 feeds simultaneously).
_API_SEMAPHORE = threading.Semaphore(3)
# Lazy-initialised singleton — created once on first call.
_client = None


class GeminiError(Exception):
    """Raised when the Gemini API call fails."""


def _get_client():
    """Return a cached ``genai.Client``, creating it on first use."""
    global _client
    if _client is not None:
        return _client

    try:
        from google import genai
    except ImportError:
        raise GeminiError("google-genai is not installed. Run: pip install google-genai") from None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiError(
            "GEMINI_API_KEY environment variable not set. "
            "Get a key at https://aistudio.google.com/apikey and add it to your .env file."
        )

    _client = genai.Client(api_key=api_key)
    return _client


def _call_api(client, model: str, prompt: str, timeout: int) -> str:
    """Make a single Gemini API call and return the response text.

    Raises the underlying SDK exception on failure (caller handles retries).
    """
    from google import genai

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.0,
            http_options=genai.types.HttpOptions(timeout=timeout * 1000),
        ),
    )
    text = response.text if response.text else ""
    if not text:
        raise GeminiError(f"Empty response from Gemini (model={model})")
    return text


def generate(prompt: str, model: str, timeout: int) -> str:
    """Call the Gemini API and return the response text.

    Args:
        prompt:  The full prompt string to send.
        model:   Gemini model name (e.g. "gemini-2.0-flash").
        timeout: Request timeout in seconds.

    Returns:
        The model's response text.

    Raises:
        GeminiError: On missing API key, connection failure, blocked content,
                     timeout, or empty response.
    """
    try:
        from google.genai import errors as _genai_errors
    except ImportError:
        raise GeminiError("google-genai is not installed. Run: pip install google-genai") from None

    client = _get_client()
    logger.debug("Gemini call | model=%s | timeout=%ds", model, timeout)

    acquired = _API_SEMAPHORE.acquire(timeout=60)
    if not acquired:
        raise GeminiError("Gemini concurrency gate timed out — 3 calls already in-flight after 60s")

    try:
        return _call_api(client, model, prompt, timeout)
    except _genai_errors.ClientError as exc:
        err_str = str(exc)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            match = re.search(r"retry.*?(\d+)s", err_str, re.IGNORECASE)
            delay = int(match.group(1)) + 1 if match else 10
            logger.warning("Gemini 429 — waiting %ds before retry", delay)
            time.sleep(delay)
            try:
                return _call_api(client, model, prompt, timeout)
            except _genai_errors.ClientError as retry_exc:
                raise GeminiError(f"Gemini API error after retry: {retry_exc}") from retry_exc
        raise GeminiError(f"Gemini API error: {exc}") from exc
    except _genai_errors.ServerError as exc:
        err_str = str(exc)
        if "503" in err_str or "UNAVAILABLE" in err_str:
            logger.warning("Gemini 503 — waiting 10s before retry")
            time.sleep(10)
            try:
                return _call_api(client, model, prompt, timeout)
            except (_genai_errors.ClientError, _genai_errors.ServerError) as retry_exc:
                raise GeminiError(f"Gemini server error after retry: {retry_exc}") from retry_exc
        raise GeminiError(f"Gemini server error: {exc}") from exc
    except GeminiError:
        raise
    except Exception as exc:
        err_str = str(exc).lower()
        if "deadline" in err_str or "timeout" in err_str:
            raise GeminiError(f"Timed out after {timeout}s (model={model})") from None
        raise GeminiError(f"Unexpected error calling Gemini: {exc}") from exc
    finally:
        _API_SEMAPHORE.release()
