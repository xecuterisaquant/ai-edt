"""Pipeline tests — no GPU or running Ollama instance required.

All calls to ai_edt.ollama.generate() are mocked. Tests cover:
  - Stage 1 keyword filtering (VIP Pass, general, no-match)
  - Stage 2 sieve behaviour (YES / NO / ambiguous / error)
  - Stage 3 signal parsing (valid, malformed, DeepSeek <think> blocks)
  - Signal history logging
  - Regression test for the original NameError bug on VIP Pass path
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from ai_edt import pipeline
from ai_edt.ollama import OllamaError
from ai_edt.signals import Signal
from tests.conftest import (
    GENERAL_KEYWORD_HEADLINES,
    HIGH_ALPHA_HEADLINES,
    IRRELEVANT_HEADLINES,
    MALFORMED_SIGNAL_RESPONSE,
    SIEVE_AMBIGUOUS_RESPONSE,
    SIEVE_NO_RESPONSE,
    SIEVE_YES_RESPONSE,
    VALID_SIGNAL_RESPONSE,
    VALID_SIGNAL_WITH_THINK,
)

# ---------------------------------------------------------------------------
# Stage 1: Keyword filter
# ---------------------------------------------------------------------------


def test_s1_no_keyword_match_returns_none(mock_config: MagicMock) -> None:
    """Completely irrelevant headline must be dropped before any LLM call."""
    with patch("ai_edt.ollama.generate") as mock_gen:
        result = pipeline.analyze("Apple reports record iPhone sales for Q4")

    assert result is None
    mock_gen.assert_not_called()


def test_s1_vip_pass_bypasses_sieve(mock_config: MagicMock) -> None:
    """High-alpha keyword must skip Stage 2 entirely — only ONE generate call."""
    with patch("ai_edt.ollama.generate", return_value=VALID_SIGNAL_RESPONSE) as mock_gen:
        pipeline.analyze("US grants Chevron expanded license for Venezuelan heavy crude")

    assert mock_gen.call_count == 1
    # The single call must be to the reasoning model, not the sieve
    assert mock_gen.call_args.kwargs["model"] == "deepseek-r1:8b"


def test_s1_general_keyword_invokes_sieve(mock_config: MagicMock) -> None:
    """General-only keyword must trigger Stage 2 (sieve called first)."""
    with patch("ai_edt.ollama.generate", return_value=SIEVE_NO_RESPONSE) as mock_gen:
        result = pipeline.analyze("Oil prices rise on supply concerns")

    # Stage 2 was called and said NO — result must be None
    assert result is None
    assert mock_gen.call_count == 1
    assert mock_gen.call_args.kwargs["model"] == "llama3.2:1b"


# ---------------------------------------------------------------------------
# Stage 2: Sieve behaviour
# ---------------------------------------------------------------------------


def test_s2_yes_proceeds_to_stage3(mock_config: MagicMock) -> None:
    """Sieve YES response must pass the headline to Stage 3."""
    responses = iter([SIEVE_YES_RESPONSE, VALID_SIGNAL_RESPONSE])
    with patch("ai_edt.ollama.generate", side_effect=responses):
        result = pipeline.analyze("Tanker rates climb as demand increases")

    assert result is not None
    assert isinstance(result, Signal)


def test_s2_no_blocks_headline(mock_config: MagicMock) -> None:
    """Sieve NO response must stop the pipeline."""
    with patch("ai_edt.ollama.generate", return_value=SIEVE_NO_RESPONSE):
        result = pipeline.analyze("Oil prices flat in quiet Tuesday trading")

    assert result is None


def test_s2_ambiguous_response_passes_through(mock_config: MagicMock) -> None:
    """Conservative filter: ambiguous sieve output must NOT block the headline."""
    responses = iter([SIEVE_AMBIGUOUS_RESPONSE, VALID_SIGNAL_RESPONSE])
    with patch("ai_edt.ollama.generate", side_effect=responses):
        result = pipeline.analyze("Tanker rates climb as demand increases")

    assert result is not None


def test_s2_ollama_error_falls_through_to_stage3(mock_config: MagicMock) -> None:
    """Sieve OllamaError must be non-fatal — pipeline proceeds to Stage 3."""
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OllamaError("Connection refused")
        return VALID_SIGNAL_RESPONSE

    with patch("ai_edt.ollama.generate", side_effect=side_effect):
        result = pipeline.analyze("Oil prices rise on supply concerns")

    assert result is not None
    assert call_count == 2


# ---------------------------------------------------------------------------
# Stage 3: Signal parsing
# ---------------------------------------------------------------------------


def test_s3_valid_signal_parsed_correctly(mock_config: MagicMock) -> None:
    """Stage 3 must parse all four fields from a well-formed response."""
    with patch("ai_edt.ollama.generate", return_value=VALID_SIGNAL_RESPONSE):
        result = pipeline.analyze("New sanctions hit Iranian tanker fleet")

    assert result is not None
    assert result.ticker == "FRO"
    assert result.direction == "LONG"
    assert result.confidence == 88
    assert len(result.rationale) > 0


def test_s3_think_tags_stripped_before_parsing(mock_config: MagicMock) -> None:
    """DeepSeek-R1 <think> chain-of-thought blocks must be stripped cleanly."""
    with patch("ai_edt.ollama.generate", return_value=VALID_SIGNAL_WITH_THINK):
        result = pipeline.analyze("New sanctions hit Iranian tanker fleet")

    assert result is not None
    assert result.ticker == "FRO"
    assert result.direction == "LONG"


def test_s3_malformed_response_returns_none(mock_config: MagicMock) -> None:
    """A response missing Ticker/Signal must return None (not raise)."""
    with patch("ai_edt.ollama.generate", return_value=MALFORMED_SIGNAL_RESPONSE):
        result = pipeline.analyze("New sanctions hit Iranian tanker fleet")

    assert result is None


# ---------------------------------------------------------------------------
# Regression: VIP Pass + Stage 3 — the original NameError bug
# ---------------------------------------------------------------------------


def test_vip_pass_does_not_raise_name_error(mock_config: MagicMock) -> None:
    """Regression: VIP Pass path must NOT raise NameError on Stage 3 Ollama call.

    In the original code, `url` was defined inside the Stage 2 else-block and
    then referenced in Stage 3, causing:
        NameError: cannot access local variable 'url' where it is not associated
        with a value
    when any high-alpha keyword triggered the VIP Pass.
    """
    with patch("ai_edt.ollama.generate", return_value=VALID_SIGNAL_RESPONSE):
        # Must not raise
        result = pipeline.analyze(
            "US grants Chevron expanded license for Venezuelan heavy crude"
        )

    assert result is not None


# ---------------------------------------------------------------------------
# Signal history logging
# ---------------------------------------------------------------------------


def test_signal_appended_to_jsonl(mock_config: MagicMock) -> None:
    """A successful Stage 3 result must be persisted to the JSONL log."""
    with patch("ai_edt.ollama.generate", return_value=VALID_SIGNAL_RESPONSE):
        signal = pipeline.analyze("New sanctions hit Iranian tanker fleet")

    assert signal is not None

    log_path = mock_config.signal_log_path
    assert log_path.exists(), "signals.jsonl was not created"

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, f"Expected 1 log entry, got {len(lines)}"

    record = json.loads(lines[0])
    assert record["ticker"] == "FRO"
    assert record["direction"] == "LONG"
    assert record["confidence"] == 88
    assert "timestamp" in record
    assert "headline" in record


def test_multiple_signals_appended_sequentially(mock_config: MagicMock) -> None:
    """Each signal must be appended as a new line — not overwriting existing data."""
    with patch("ai_edt.ollama.generate", return_value=VALID_SIGNAL_RESPONSE):
        pipeline.analyze("New sanctions hit Iranian tanker fleet")
        pipeline.analyze("Suez Canal closure forces rerouting of oil tankers")

    log_path = mock_config.signal_log_path
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    # Each line must be valid JSON
    for line in lines:
        json.loads(line)
