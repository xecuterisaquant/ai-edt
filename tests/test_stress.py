#!/usr/bin/env python3
"""AI-EDT sieve stress-test suite.

Runs headlines through the live pipeline and scores accuracy.
Not part of CI (requires Gemini API key and network); run manually:

    python -m pytest tests/test_stress.py -s --tb=short

Or standalone:

    python tests/test_stress.py
"""

from __future__ import annotations

import dataclasses
import time

from ai_edt import pipeline
from ai_edt.logger import get_logger
from ai_edt.pipeline import stage1_matches
from ai_edt.signals import Signal

logger = get_logger("stress")

# ---------------------------------------------------------------------------
# Sieve stress-test suite  (~33 headlines, 5 groups)
#
#   GROUP A — Sieve should BLOCK (8 headlines)
#   GROUP B — Sieve should PASS with expected signal (8 headlines)
#   GROUP C — Stage 1 BLIND SPOTS (5 headlines)
#   GROUP D — SHORT scenarios (2 headlines)
#   GROUP E — Full KB coverage (10 headlines)
# ---------------------------------------------------------------------------

TEST_GROUPS = [
    {
        "label": "GROUP A — Sieve should BLOCK (noise with keyword match)",
        "expected": "skip",
        "headlines": [
            {"text": "Oil company CEO resigns amid accounting scandal"},
            {"text": "Oil market analysts forecast stable prices through summer"},
            {"text": "Vessel docks at Rotterdam after routine transoceanic voyage"},
            {"text": "'Fleet of the Future' tech conference opens in Singapore"},
            {"text": "Refinery workers union votes to accept new pay deal"},
            {"text": "Oil company reports quarterly dividend within expectations"},
            {"text": "Tanker company launches new sustainability report"},
            {"text": "Scheduled crude delivery completes — vessel departs Ras Tanura on schedule"},
        ],
    },
    {
        "label": "GROUP B — Sieve should PASS (expected signal: ticker + direction)",
        "expected": "signal",
        "headlines": [
            {
                "text": "Refinery explosion at Motiva Port Arthur forces unplanned 3-week outage",
                "expected_ticker": "PBF",
                "expected_direction": "LONG",
            },
            {
                "text": "OPEC+ agrees surprise 500,000 bpd production cut starting April",
                "expected_ticker": None,
                "expected_direction": "LONG",
            },
            {
                "text": "Tanker rates on VLCC routes hit 6-month high amid tightening cargo supply",
                "expected_ticker": "FRO",
                "expected_direction": "LONG",
            },
            {
                "text": "US Treasury sanctions 20 shadow fleet tankers in latest enforcement action",
                "expected_ticker": "FRO",
                "expected_direction": "LONG",
            },
            {
                "text": "Heavy sour crude discount to Brent widens sharply to $8/bbl",
                "expected_ticker": "PBF",
                "expected_direction": "LONG",
            },
            {
                "text": "Libya Sharara oilfield shuts — 300,000 bpd offline, workers evacuated",
                "expected_ticker": None,
                "expected_direction": "LONG",
            },
            {
                "text": "Venezuelan crude tanker bookings surge on expanded US export licences",
                "expected_ticker": "TNK",
                "expected_direction": "LONG",
            },
            {
                "text": "VLCC spot day rates collapse 40% as surge of newbuild deliveries hits market",
                "expected_ticker": "FRO",
                "expected_direction": "SHORT",
            },
        ],
    },
    {
        "label": "GROUP C — Stage 1 BLIND SPOTS (trade-relevant, zero keyword match)",
        "expected": "miss — Stage 1 gap",
        "headlines": [
            {"text": "India doubles strategic petroleum reserve purchases"},
            {"text": "Panama Canal halts Neopanamax transits — low water drought restrictions"},
            {"text": "Saudi Arabia raises crude OSP for Asia-bound grades by $2/bbl"},
            {"text": "Drone strike reported near Saudi Ras Tanura terminal — loading suspended"},
            {"text": "Caspian Pipeline Consortium resumes exports after week-long suspension"},
        ],
    },
    {
        "label": "GROUP D — SHORT scenarios (fleet oversupply / rate collapse)",
        "expected": "signal",
        "headlines": [
            {
                "text": "Global VLCC fleet grows 5% as massive newbuild wave enters service",
                "expected_ticker": "FRO",
                "expected_direction": "SHORT",
            },
            {
                "text": "Baltic Exchange: VLCC freight rates hit 18-month low on tonnage oversupply",
                "expected_ticker": "FRO",
                "expected_direction": "SHORT",
            },
        ],
    },
    {
        "label": "GROUP E — Full KB coverage (tickers not tested in A–D)",
        "expected": "signal",
        "headlines": [
            {
                "text": "Marathon Garyville refinery declares force majeure after flooding — 560,000 bpd offline",
                "expected_ticker": "VLO",
                "expected_direction": "LONG",
            },
            {
                "text": "WTI crude discount to Brent widens to $7/bbl on Cushing inventory build",
                "expected_ticker": "MPC",
                "expected_direction": "LONG",
            },
            {
                "text": "US LPG export volumes hit record high as Gulf Coast fractionation utilisation surges",
                "expected_ticker": "PSX",
                "expected_direction": "LONG",
            },
            {
                "text": "Brent crude surges $12/bbl after surprise drawdown at Cushing and Fujairah",
                "expected_ticker": "PBF",
                "expected_direction": "SHORT",
            },
            {
                "text": "Keystone Pipeline system shut after rupture detected in South Dakota — indefinite closure",
                "expected_ticker": "ENB",
                "expected_direction": "LONG",
            },
            {
                "text": "Trans-Atlantic clean tanker freight rates surge to 3-year high as US diesel fleet fully booked for European delivery",
                "expected_ticker": "STNG",
                "expected_direction": "LONG",
            },
            {
                "text": "Suezmax spot rates hit 2-year high as Red Sea diversions absorb 18% of fleet",
                "expected_ticker": "TNK",
                "expected_direction": "LONG",
            },
            {
                "text": "VLCC time-charter rates fall sharply as Asian refiners cut crude import programmes",
                "expected_ticker": "FRO",
                "expected_direction": "SHORT",
            },
            {
                "text": "Strait of Hormuz partially blocked — crude and product tanker diversions underway",
                "expected_ticker": None,
                "expected_direction": "LONG",
            },
            {
                "text": "Permian Basin output reaches 7 million bpd — highest ever as shale productivity climbs",
                "expected_ticker": "MPC",
                "expected_direction": "LONG",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Accuracy tracking
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Stats:
    correct_signals: list[str] = dataclasses.field(default_factory=list)
    wrong_ticker: list[str] = dataclasses.field(default_factory=list)
    wrong_direction: list[str] = dataclasses.field(default_factory=list)
    wrong_both: list[str] = dataclasses.field(default_factory=list)
    false_positives: list[str] = dataclasses.field(default_factory=list)
    false_negatives: list[str] = dataclasses.field(default_factory=list)
    blind_spots: list[str] = dataclasses.field(default_factory=list)


def _stage1_would_pass(headline_lower: str) -> bool:
    """Return True if Stage 1 would forward this headline to Stage 2/3."""
    tier = stage1_matches(headline_lower)
    return tier in ("vip", "general")


def _handle_signal(
    stats: _Stats,
    headline: str,
    signal: Signal,
    expected: str,
    exp_ticker: str | None,
    exp_direction: str | None,
) -> None:
    """Classify and log a headline that produced a signal."""
    logger.info(
        "  Result   : SIGNAL → %s %s @ %d%%",
        signal.ticker,
        signal.direction,
        signal.confidence,
    )
    logger.info("  Rationale: %s", signal.rationale)

    if expected == "skip":
        stats.false_positives.append(headline)
        logger.warning("  ⚠ FALSE POSITIVE — sieve let noise through to 8B")
        return

    if expected == "miss — Stage 1 gap":
        logger.info("  ℹ STAGE 1 COVERAGE — blind spot now captured by keywords")
        return

    # expected == "signal"
    ticker_ok = exp_ticker is None or signal.ticker == exp_ticker
    dir_ok = exp_direction is None or signal.direction == exp_direction

    if ticker_ok and dir_ok:
        stats.correct_signals.append(headline)
        logger.info(
            "  ✓ CORRECT   — %s %s (expected: %s %s)",
            signal.ticker,
            signal.direction,
            exp_ticker or "any",
            exp_direction or "any",
        )
    elif dir_ok:
        stats.wrong_ticker.append(headline)
        logger.warning("  ⚠ WRONG TICKER — got %s, expected %s", signal.ticker, exp_ticker)
    elif ticker_ok:
        stats.wrong_direction.append(headline)
        logger.warning(
            "  ⚠ WRONG DIRECTION — got %s, expected %s",
            signal.direction,
            exp_direction,
        )
    else:
        stats.wrong_both.append(headline)
        logger.warning(
            "  ⚠ WRONG — got %s %s, expected %s %s",
            signal.ticker,
            signal.direction,
            exp_ticker,
            exp_direction,
        )


def _handle_no_signal(
    stats: _Stats,
    headline: str,
    expected: str,
    stage1_passed: bool,
) -> None:
    """Classify and log a headline that produced no signal."""
    logger.info("  Result   : No signal generated")

    if expected == "signal":
        stats.false_negatives.append(headline)
        if stage1_passed:
            logger.warning("  ⚠ FALSE NEGATIVE — pipeline reached S3 but timed out or parse failed")
        else:
            logger.warning("  ⚠ FALSE NEGATIVE — Stage 1 never matched (add keyword)")

    elif expected == "miss — Stage 1 gap":
        if not stage1_passed:
            stats.blind_spots.append(headline)
            logger.warning("  ⚠ BLIND SPOT — Stage 1 never saw this headline")
        else:
            logger.warning("  ⚠ STAGE 3 TIMEOUT — headline reached S3 but no result")


def _warn_list(items: list[str], label: str, description: str) -> None:
    if not items:
        return
    logger.warning("%s (%d) — %s:", label, len(items), description)
    for h in items:
        logger.warning("  • %s", h)


def _log_summary(stats: _Stats) -> None:
    judged = (
        len(stats.correct_signals)
        + len(stats.wrong_ticker)
        + len(stats.wrong_direction)
        + len(stats.wrong_both)
    )
    accuracy_pct = round(100 * len(stats.correct_signals) / judged) if judged else 0
    noise_count = sum(len(g["headlines"]) for g in TEST_GROUPS if g["expected"] == "skip")
    noise_blocked = noise_count - len(stats.false_positives)

    logger.info("")
    logger.info("=" * 64)
    logger.info("STRESS TEST SUMMARY")
    logger.info("=" * 64)
    logger.info(
        "Signal accuracy : %d/%d correct ticker+direction  (%d%%)",
        len(stats.correct_signals),
        judged,
        accuracy_pct,
    )
    logger.info(
        "Filter accuracy : %d/%d noise headlines correctly blocked",
        noise_blocked,
        noise_count,
    )
    _warn_list(stats.false_positives, "FALSE POSITIVES", "noise that reached the reasoning engine")
    _warn_list(stats.false_negatives, "FALSE NEGATIVES", "real trades the pipeline missed")
    _warn_list(stats.wrong_ticker, "WRONG TICKER", "correct direction, wrong company")
    _warn_list(stats.wrong_direction, "WRONG DIRECTION", "correct ticker, wrong LONG/SHORT call")
    _warn_list(stats.wrong_both, "WRONG BOTH", "wrong ticker AND wrong direction")
    _warn_list(stats.blind_spots, "BLIND SPOTS", "add keywords to config/keywords.yaml")

    all_clean = not any(
        [
            stats.false_positives,
            stats.false_negatives,
            stats.wrong_ticker,
            stats.wrong_direction,
            stats.wrong_both,
            stats.blind_spots,
        ]
    )
    if all_clean:
        logger.info("All headlines behaved as expected.")

    logger.info("")
    logger.info("Full debug log at logs/ai_edt.log")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_stress_test() -> None:
    """Run the stress-test suite and report accuracy metrics."""
    stats = _Stats()
    prev_entered_pipeline = False

    for group in TEST_GROUPS:
        logger.info("")
        logger.info("=" * 64)
        logger.info(group["label"])
        logger.info("=" * 64)

        for entry in group["headlines"]:
            headline: str = entry["text"]
            exp_ticker: str | None = entry.get("expected_ticker")
            exp_direction: str | None = entry.get("expected_direction")

            logger.info("")
            logger.info("  Headline : %s", headline)

            if prev_entered_pipeline:
                time.sleep(5)  # GPU cooldown — reduces thermal throttle timeouts

            stage1_passed = _stage1_would_pass(headline.lower())
            signals = pipeline.analyze(headline)
            prev_entered_pipeline = stage1_passed

            # Use the highest-confidence (first) signal for accuracy scoring.
            signal = signals[0] if signals else None

            if signal:
                _handle_signal(
                    stats,
                    headline,
                    signal,
                    group["expected"],
                    exp_ticker,
                    exp_direction,
                )
            else:
                _handle_no_signal(stats, headline, group["expected"], stage1_passed)

    _log_summary(stats)


if __name__ == "__main__":
    run_stress_test()
