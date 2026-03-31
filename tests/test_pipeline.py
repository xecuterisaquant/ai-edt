"""Pipeline tests — no GPU or running Ollama instance required.

All calls to ai_edt.ollama.generate() are mocked. Tests cover:
  - Stage 1 keyword filtering (VIP Pass, general, no-match, no_pass block)
  - Stage 2 sieve behaviour (YES / NO / ambiguous / error)
  - Stage 3 signal parsing (valid, malformed, DeepSeek <think> blocks, SHORT)
  - Multi-order signal parsing (Phase 5: up to 3 ranked signals per headline)
  - Sector-aware KB filtering (additive union across matched sectors)
  - Duplicate signal detection
  - Signal history logging
  - Regression test for the original NameError bug on VIP Pass path
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

import ai_edt.db as db_mod
from ai_edt import pipeline
from ai_edt.ollama import OllamaError
from ai_edt.signals import Signal, is_duplicate_signal
from tests.conftest import (
    MALFORMED_SIGNAL_RESPONSE,
    MULTI_SIGNAL_DUPLICATE_TICKER,
    SIEVE_AMBIGUOUS_RESPONSE,
    SIEVE_NO_RESPONSE,
    SIEVE_YES_RESPONSE,
    VALID_MULTI_SIGNAL_RESPONSE,
    VALID_MULTI_SIGNAL_TWO_ONLY,
    VALID_MULTI_SIGNAL_WITH_THINK,
    VALID_SHORT_SIGNAL_RESPONSE,
    VALID_SIGNAL_RESPONSE,
    VALID_SIGNAL_WITH_THINK,
    VALID_TNK_SIGNAL_RESPONSE,
)

# ---------------------------------------------------------------------------
# Stage 1: Keyword filter
# ---------------------------------------------------------------------------


def test_s1_no_keyword_match_returns_none(mock_config: MagicMock) -> None:
    """Completely irrelevant headline must be dropped before any LLM call."""
    with patch("ai_edt.ollama.generate") as mock_gen:
        result = pipeline.analyze("Apple reports record iPhone sales for Q4")

    assert result == []
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

    # Stage 2 was called and said NO — result must be empty list
    assert result == []
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

    assert len(result) >= 1
    assert isinstance(result[0], Signal)


def test_s2_no_blocks_headline(mock_config: MagicMock) -> None:
    """Sieve NO response must stop the pipeline."""
    with patch("ai_edt.ollama.generate", return_value=SIEVE_NO_RESPONSE):
        result = pipeline.analyze("Oil prices flat in quiet Tuesday trading")

    assert result == []


def test_s2_ambiguous_response_passes_through(mock_config: MagicMock) -> None:
    """Conservative filter: ambiguous sieve output must NOT block the headline."""
    responses = iter([SIEVE_AMBIGUOUS_RESPONSE, VALID_SIGNAL_RESPONSE])
    with patch("ai_edt.ollama.generate", side_effect=responses):
        result = pipeline.analyze("Tanker rates climb as demand increases")

    assert len(result) >= 1


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

    assert len(result) >= 1
    assert call_count == 2


# ---------------------------------------------------------------------------
# Stage 3: Signal parsing
# ---------------------------------------------------------------------------


def test_s3_valid_signal_parsed_correctly(mock_config: MagicMock) -> None:
    """Stage 3 must parse all four fields from a well-formed legacy response."""
    with patch("ai_edt.ollama.generate", return_value=VALID_SIGNAL_RESPONSE):
        result = pipeline.analyze("New sanctions hit Iranian tanker fleet")

    assert len(result) == 1
    sig = result[0]
    assert sig.ticker == "FRO"
    assert sig.direction == "LONG"
    assert sig.confidence == 88
    assert len(sig.rationale) > 0


def test_s3_think_tags_stripped_before_parsing(mock_config: MagicMock) -> None:
    """DeepSeek-R1 <think> chain-of-thought blocks must be stripped cleanly."""
    with patch("ai_edt.ollama.generate", return_value=VALID_SIGNAL_WITH_THINK):
        result = pipeline.analyze("New sanctions hit Iranian tanker fleet")

    assert len(result) >= 1
    assert result[0].ticker == "FRO"
    assert result[0].direction == "LONG"


def test_s3_malformed_response_returns_none(mock_config: MagicMock) -> None:
    """A response missing all recognisable fields must return empty list."""
    with patch("ai_edt.ollama.generate", return_value=MALFORMED_SIGNAL_RESPONSE):
        result = pipeline.analyze("New sanctions hit Iranian tanker fleet")

    assert result == []


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
        result = pipeline.analyze("US grants Chevron expanded license for Venezuelan heavy crude")

    assert len(result) >= 1


# ---------------------------------------------------------------------------
# Signal persistence (SQLite)
# ---------------------------------------------------------------------------


def test_signal_inserted_to_db(mock_config: MagicMock) -> None:
    """A successful Stage 3 result must be persisted to the SQLite database."""
    with patch("ai_edt.ollama.generate", return_value=VALID_SIGNAL_RESPONSE):
        signals = pipeline.analyze("New sanctions hit Iranian tanker fleet")

    assert len(signals) == 1

    rows = db_mod.get_recent_signals(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["ticker"] == "FRO"
    assert row["direction"] == "LONG"
    assert row["confidence"] == 88
    assert row["headline"] == "New sanctions hit Iranian tanker fleet"
    assert row["created_utc"] is not None


def test_feed_source_stored_with_signal(mock_config: MagicMock) -> None:
    """feed_source passed to analyze() must be stored in the database row."""
    with patch("ai_edt.ollama.generate", return_value=VALID_SIGNAL_RESPONSE):
        pipeline.analyze(
            "New sanctions hit Iranian tanker fleet",
            feed_source="Reuters Energy",
        )

    rows = db_mod.get_recent_signals(limit=10)
    assert rows[0]["feed_source"] == "Reuters Energy"


def test_different_ticker_signals_both_logged(mock_config: MagicMock) -> None:
    """Signals for different tickers must both be persisted."""
    responses = iter([VALID_SIGNAL_RESPONSE, VALID_TNK_SIGNAL_RESPONSE])
    with patch("ai_edt.ollama.generate", side_effect=responses):
        pipeline.analyze("New sanctions hit Iranian tanker fleet")
        pipeline.analyze("Venezuelan crude volumes increase on Caribbean route")

    rows = db_mod.get_recent_signals(limit=10)
    assert len(rows) == 2
    tickers = {r["ticker"] for r in rows}
    assert tickers == {"FRO", "TNK"}


def test_duplicate_ticker_direction_suppressed(mock_config: MagicMock) -> None:
    """Same ticker+direction within 30 minutes must only be persisted once."""
    with patch("ai_edt.ollama.generate", return_value=VALID_SIGNAL_RESPONSE):
        pipeline.analyze("New sanctions hit Iranian tanker fleet")
        pipeline.analyze("Suez Canal closure forces rerouting of oil tankers")

    # Both produce FRO LONG — second must be suppressed as a duplicate.
    rows = db_mod.get_recent_signals(limit=10)
    assert len(rows) == 1, "Duplicate FRO LONG within 30-minute window should be suppressed"


# ---------------------------------------------------------------------------
# Stage 1: no_pass keyword blocking
# ---------------------------------------------------------------------------


def test_s1_no_pass_keyword_blocks_without_llm(mock_config: MagicMock) -> None:
    """A no_pass keyword must return empty list immediately, before any LLM call."""
    with patch("ai_edt.ollama.generate") as mock_gen:
        result = pipeline.analyze("Oil company CEO resigns amid accounting scandal")

    assert result == []
    mock_gen.assert_not_called()


@pytest.mark.parametrize(
    "headline",
    [
        "Oil company reports quarterly dividend within expectations",
        "Tanker company launches new sustainability report",
        "Refiner releases annual report with record revenue",
        "Oil major holds investor day for long-term strategy update",
    ],
)
def test_s1_no_pass_corporate_comms_blocks_without_llm(
    headline: str, mock_config: MagicMock
) -> None:
    """Corporate calendar and comms headlines must be blocked at Stage 1."""
    with patch("ai_edt.ollama.generate") as mock_gen:
        result = pipeline.analyze(headline)

    assert result == []
    mock_gen.assert_not_called()


# ---------------------------------------------------------------------------
# Stage 3: SHORT signal direction
# ---------------------------------------------------------------------------


def test_s3_short_signal_parsed_correctly(mock_config: MagicMock) -> None:
    """Stage 3 must parse a SHORT direction correctly from the model response."""
    with patch("ai_edt.ollama.generate", return_value=VALID_SHORT_SIGNAL_RESPONSE):
        result = pipeline.analyze("New sanctions hit Iranian tanker fleet")

    assert len(result) >= 1
    assert result[0].ticker == "FRO"
    assert result[0].direction == "SHORT"
    assert result[0].confidence == 80


# ---------------------------------------------------------------------------
# Sector-aware KB filtering
# ---------------------------------------------------------------------------


def test_sector_filter_logistics_headline_excludes_refinery_data(
    mock_config: MagicMock,
) -> None:
    """A logistics-only headline must NOT include refinery_data in the Stage 3 prompt.

    'sanction' is in mock high_alpha → VIP pass (single Ollama call).
    'iran' matches _LOGISTICS_HINTS → sector filter strips refinery_data.
    Checked by verifying the company name 'PBF Energy' is absent from the prompt.
    """
    with patch("ai_edt.ollama.generate", return_value=VALID_SIGNAL_RESPONSE) as mock_gen:
        pipeline.analyze("New sanctions hit Iranian tanker fleet")

    prompt_sent = mock_gen.call_args.kwargs["prompt"]
    assert "PBF Energy" not in prompt_sent, "Refinery data leaked into logistics prompt"
    assert "Frontline" in prompt_sent, "Logistics data missing from logistics prompt"


def test_sector_filter_refinery_headline_excludes_logistics_data(
    mock_config: MagicMock,
) -> None:
    """A refinery-only headline must NOT include logistics_data in the Stage 3 prompt.

    'heavy crude' is in mock high_alpha → VIP pass.
    'refinery' and 'crack' match _REFINERY_HINTS → sector filter strips logistics_data.
    Checked by verifying 'Frontline PLC' is absent from the prompt.
    """
    with patch("ai_edt.ollama.generate", return_value=VALID_SIGNAL_RESPONSE) as mock_gen:
        pipeline.analyze("Motiva refinery crack spread widens on heavy crude discount")

    prompt_sent = mock_gen.call_args.kwargs["prompt"]
    assert "Frontline" not in prompt_sent, "Logistics data leaked into refinery prompt"
    assert "PBF Energy" in prompt_sent, "Refinery data missing from refinery prompt"


def test_sector_filter_multi_sector_includes_both(
    mock_config: MagicMock,
) -> None:
    """A headline matching both shipping and refinery hints must include both sections.

    The additive KB filter (Phase 5) should include logistics AND refinery data
    when both sector hint sets fire, giving Stage 3 full cross-sector context.
    """
    with patch("ai_edt.ollama.generate", return_value=VALID_SIGNAL_RESPONSE) as mock_gen:
        pipeline.analyze("Iran sanctions hit tanker fleet and crack spreads in refinery sector")

    prompt_sent = mock_gen.call_args.kwargs["prompt"]
    assert "Frontline" in prompt_sent, "Shipping data missing from multi-sector prompt"
    assert "PBF Energy" in prompt_sent, "Refinery data missing from multi-sector prompt"


# ---------------------------------------------------------------------------
# Duplicate signal detection
# ---------------------------------------------------------------------------


def test_is_duplicate_signal_within_window(mock_config: MagicMock) -> None:
    """Same ticker+direction inserted into DB must be flagged as a duplicate."""
    existing = Signal(
        headline="Iran sanctions",
        ticker="FRO",
        direction="LONG",
        confidence=90,
        rationale="existing",
    )
    db_mod.insert_signal(existing)

    new = Signal(
        headline="Red Sea attack",
        ticker="FRO",
        direction="LONG",
        confidence=85,
        rationale="new",
    )
    assert is_duplicate_signal(new, window_minutes=30) is True


def test_is_duplicate_signal_outside_window(mock_config: MagicMock) -> None:
    """A signal outside the time window must NOT be flagged as a duplicate."""
    old_ts = (datetime.now(UTC) - timedelta(minutes=60)).isoformat()
    existing = Signal(
        headline="old event",
        ticker="FRO",
        direction="LONG",
        confidence=90,
        rationale="old",
        timestamp=old_ts,
    )
    db_mod.insert_signal(existing)

    new = Signal(
        headline="new event",
        ticker="FRO",
        direction="LONG",
        confidence=85,
        rationale="new",
    )
    assert is_duplicate_signal(new, window_minutes=30) is False


def test_is_duplicate_signal_different_ticker(mock_config: MagicMock) -> None:
    """Different tickers must never be considered duplicates."""
    existing = Signal(
        headline="sanctions",
        ticker="FRO",
        direction="LONG",
        confidence=90,
        rationale="existing",
    )
    db_mod.insert_signal(existing)

    new = Signal(
        headline="related event",
        ticker="TNK",
        direction="LONG",
        confidence=85,
        rationale="new",
    )
    assert is_duplicate_signal(new, window_minutes=30) is False


# ---------------------------------------------------------------------------
# Phase 5: Multi-order signal parsing
# ---------------------------------------------------------------------------


def test_multi_signal_three_parsed(mock_config: MagicMock) -> None:
    """Three-signal response must produce 3 Signal objects, ranked in order."""
    with patch("ai_edt.ollama.generate", return_value=VALID_MULTI_SIGNAL_RESPONSE):
        result = pipeline.analyze("New sanctions hit Iranian tanker fleet")

    assert len(result) == 3
    assert result[0].ticker == "FRO"
    assert result[0].direction == "LONG"
    assert result[0].confidence == 95
    assert result[0].order_level == 1
    assert result[1].ticker == "EURN"
    assert result[1].confidence == 80
    assert result[1].order_level == 2
    assert result[2].ticker == "INSW"
    assert result[2].confidence == 70
    assert result[2].order_level == 3


def test_multi_signal_two_only(mock_config: MagicMock) -> None:
    """A two-signal response must produce exactly 2 Signal objects."""
    with patch("ai_edt.ollama.generate", return_value=VALID_MULTI_SIGNAL_TWO_ONLY):
        result = pipeline.analyze("Motiva refinery fire — 600,000 bpd offline")

    assert len(result) == 2
    assert result[0].ticker == "PBF"
    assert result[1].ticker == "VLO"


def test_multi_signal_think_tags_stripped(mock_config: MagicMock) -> None:
    """<think> blocks inside a multi-signal response must be stripped before parsing."""
    with patch("ai_edt.ollama.generate", return_value=VALID_MULTI_SIGNAL_WITH_THINK):
        result = pipeline.analyze("New sanctions hit Iranian tanker fleet")

    assert len(result) == 2
    assert result[0].ticker == "FRO"
    assert result[1].ticker == "DHT"


def test_multi_signal_duplicate_ticker_deduplicated(mock_config: MagicMock) -> None:
    """Duplicate tickers in a multi-signal response must be deduplicated."""
    with patch("ai_edt.ollama.generate", return_value=MULTI_SIGNAL_DUPLICATE_TICKER):
        result = pipeline.analyze("New sanctions hit Iranian tanker fleet")

    tickers = [s.ticker for s in result]
    assert tickers.count("FRO") == 1, "Duplicate FRO ticker must be deduplicated"
    assert "EURN" in tickers


def test_multi_signal_all_share_event_id(mock_config: MagicMock) -> None:
    """All signals from the same headline must share the same event_id."""
    with patch("ai_edt.ollama.generate", return_value=VALID_MULTI_SIGNAL_RESPONSE):
        result = pipeline.analyze("New sanctions hit Iranian tanker fleet")

    assert len(result) == 3
    event_ids = {s.event_id for s in result}
    assert len(event_ids) == 1, "All signals from same headline must share event_id"
    assert event_ids.pop() != ""


def test_multi_signal_event_id_differs_between_headlines(mock_config: MagicMock) -> None:
    """Different headlines must produce different event_ids."""
    with patch("ai_edt.ollama.generate", return_value=VALID_MULTI_SIGNAL_RESPONSE):
        result1 = pipeline.analyze("New sanctions hit Iranian tanker fleet")
        result2 = pipeline.analyze("Suez Canal closure forces rerouting")

    assert result1[0].event_id != result2[0].event_id


def test_multi_signal_all_logged_to_db(mock_config: MagicMock) -> None:
    """All signals in a multi-signal response must be persisted to the DB."""
    with patch("ai_edt.ollama.generate", return_value=VALID_MULTI_SIGNAL_RESPONSE):
        pipeline.analyze("New sanctions hit Iranian tanker fleet")

    rows = db_mod.get_recent_signals(limit=10)
    assert len(rows) == 3
    tickers = {r["ticker"] for r in rows}
    assert tickers == {"FRO", "EURN", "INSW"}


def test_multi_signal_shared_event_id_in_db(mock_config: MagicMock) -> None:
    """All DB rows from a multi-order event must share the same event_id."""
    with patch("ai_edt.ollama.generate", return_value=VALID_MULTI_SIGNAL_RESPONSE):
        pipeline.analyze("New sanctions hit Iranian tanker fleet")

    rows = db_mod.get_recent_signals(limit=10)
    event_ids = {r["event_id"] for r in rows}
    assert len(event_ids) == 1
    assert None not in event_ids


def test_legacy_single_signal_still_works(mock_config: MagicMock) -> None:
    """Legacy single-signal format must still produce exactly 1 Signal."""
    with patch("ai_edt.ollama.generate", return_value=VALID_SIGNAL_RESPONSE):
        result = pipeline.analyze("New sanctions hit Iranian tanker fleet")

    assert len(result) == 1
    assert result[0].ticker == "FRO"
