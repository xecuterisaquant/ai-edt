#!/usr/bin/env python3
"""AI-EDT entry point.

Usage:
    python run.py           Sieve stress-test — runs headlines across three
                            groups to find filter failures and coverage gaps.

    python run.py watch     Live mode — starts the RSS feed watcher,
                            polling every 2 minutes for breaking news.
"""

from __future__ import annotations

import sys
import time

from ai_edt.logger import get_logger

logger = get_logger("run")

# ---------------------------------------------------------------------------
# Sieve stress-test suite
#
# Three groups designed to surface every failure mode:
#
#   GROUP A — Sieve should BLOCK
#     General keyword matches (oil / tanker / vessel / fleet / refinery / bpd)
#     but the headline carries no actionable trade information.
#     A false-positive here wastes an expensive 8B inference call.
#
#   GROUP B — Sieve should PASS
#     General keyword AND genuinely trade-relevant. Only 1B sieve runs first;
#     the 8B engine should then produce a signal.
#     A false-negative here means a real trade is missed entirely.
#
#   GROUP C — Stage 1 BLIND SPOTS  (most important)
#     Clearly trade-relevant headlines with NO keyword match at all.
#     These never reach the sieve — the whole pipeline silently skips them.
#     Every miss here is a gap to fix in config/keywords.yaml.
# ---------------------------------------------------------------------------

TEST_GROUPS = [
    {
        "label": "GROUP A — Sieve should BLOCK (noise with keyword match)",
        "expected": "skip",
        "headlines": [
            "Oil company CEO resigns amid accounting scandal",
            "Oil market analysts forecast stable prices through summer",
            "Vessel docks at Rotterdam after routine transoceanic voyage",
            "'Fleet of the Future' tech conference opens in Singapore",
            "Refinery workers union votes to accept new pay deal",
        ],
    },
    {
        "label": "GROUP B — Sieve should PASS (general keyword, actionable)",
        "expected": "signal",
        "headlines": [
            "Refinery explosion at Motiva Port Arthur forces unplanned 3-week outage",
            "OPEC+ agrees surprise 500,000 bpd production cut starting April",
            "Tanker rates on VLCC routes hit 6-month high amid tightening cargo supply",
        ],
    },
    {
        "label": "GROUP C — Stage 1 BLIND SPOTS (trade-relevant, zero keyword match)",
        "expected": "miss — Stage 1 gap",
        "headlines": [
            "Red Sea Houthi attacks resume — major shipping firms announce Cape rerouting",
            "Iran nuclear deal talks collapse, no agreement reached",
            "OPEC emergency meeting called for Thursday in Vienna amid price slide",
            "Strait of Malacca transit fees doubled following regional security tensions",
            "Yemen ceasefire collapses — Houthi drones strike two cargo ships overnight",
        ],
    },
]


def demo() -> None:
    from ai_edt import pipeline
    from ai_edt.config import get_config

    cfg = get_config()
    # Track whether the last call reached Stage 3 so we can insert a GPU
    # cooldown between consecutive expensive inferences.
    _last_call_was_s3 = False

    skipped_vip = []
    false_positives = []   # Group A: sieve passed when it should have blocked
    false_negatives = []   # Group B: sieve blocked when it should have passed
    blind_spots = []       # Group C: Stage 1 never saw it

    for group in TEST_GROUPS:
        logger.info("")
        logger.info("=" * 64)
        logger.info(group["label"])
        logger.info("=" * 64)

        for headline in group["headlines"]:
            logger.info("")
            logger.info("  Headline : %s", headline)

            # Small cooldown after Stage 3 calls to let the GPU partially
            # recover between inferences. Avoids thermal-throttle timeouts
            # when running many headlines back to back in the stress test.
            if _last_call_was_s3:
                time.sleep(4)

            # Peek at Stage 1 to detect the difference between a true blind
            # spot (Stage 1 skipped the headline) and a Stage 3 timeout that
            # returns None but passed Stage 1 correctly.
            headline_lower = headline.lower()
            all_kw = cfg.high_alpha_keywords + cfg.general_keywords
            stage1_matched = any(k in headline_lower for k in all_kw) and not any(
                k in headline_lower for k in cfg.no_pass_keywords
            )

            signal = pipeline.analyze(headline)
            _last_call_was_s3 = stage1_matched  # any Stage 1 pass = potential S3 call

            if signal:
                result_tag = f"SIGNAL → {signal.ticker} {signal.direction} @ {signal.confidence}%"
                logger.info("  Result   : %s", result_tag)
                logger.info("  Rationale: %s", signal.rationale)
                if group["expected"] == "skip":
                    false_positives.append(headline)
                    logger.warning("  ⚠ FALSE POSITIVE — sieve let noise through to 8B")
            else:
                logger.info("  Result   : No signal generated")
                if group["expected"] == "signal":
                    if stage1_matched:
                        false_negatives.append(headline)
                        logger.warning("  ⚠ FALSE NEGATIVE — sieve blocked a real trade")
                    else:
                        false_negatives.append(headline)
                        logger.warning("  ⚠ FALSE NEGATIVE — Stage 1 never matched (add keyword)")
                elif group["expected"] == "miss — Stage 1 gap":
                    if not stage1_matched:
                        blind_spots.append(headline)
                        logger.warning("  ⚠ BLIND SPOT — Stage 1 never saw this headline")
                    else:
                        # Passed Stage 1 but returned None — most likely a timeout
                        logger.warning("  ⚠ STAGE 3 TIMEOUT — headline reached S3 but no result")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 64)
    logger.info("STRESS TEST SUMMARY")
    logger.info("=" * 64)

    if false_positives:
        logger.warning("FALSE POSITIVES (%d) — sieve noise that reached 8B:", len(false_positives))
        for h in false_positives:
            logger.warning("  • %s", h)

    if false_negatives:
        logger.warning("FALSE NEGATIVES (%d) — real trades the sieve blocked:", len(false_negatives))
        for h in false_negatives:
            logger.warning("  • %s", h)

    if blind_spots:
        logger.warning("BLIND SPOTS (%d) — add keywords to config/keywords.yaml:", len(blind_spots))
        for h in blind_spots:
            logger.warning("  • %s", h)

    if not false_positives and not false_negatives and not blind_spots:
        logger.info("All headlines behaved as expected.")

    logger.info("")
    logger.info("Full debug log at logs/ai_edt.log")


def watch() -> None:
    from ai_edt import watcher

    watcher.start()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        watch()
    else:
        demo()
