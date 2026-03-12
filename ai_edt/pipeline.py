"""3-stage analysis pipeline for AI-EDT.

Stage 1 — Keyword filter    (zero cost, pure Python string matching)
           Sub-tier A: no_pass keywords  → deterministic BLOCK
           Sub-tier B: high_alpha keywords → VIP bypass straight to Stage 3
           Sub-tier C: general keywords  → proceed to Stage 2
Stage 2 — Mini-LLM sieve   (Llama-3.2-1B, skipped on VIP Pass)
Stage 3 — Reasoning engine  (DeepSeek-R1-8B, 2nd-order market analysis)

The pipeline is conservative at Stage 2: it only blocks a headline on a
clear, unambiguous "NO" from the 1B model. Ambiguous responses pass through.
This trades occasional false positives for zero missed signals.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from ai_edt import ollama
from ai_edt.config import get_config
from ai_edt.logger import get_logger
from ai_edt.ollama import OllamaError
from ai_edt.signals import Signal, log_signal, parse_signal, sieve_says_no

logger = get_logger("pipeline")

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SIEVE_PROMPT = (
    "You are a news classifier for an oil & gas trading system.\n\n"
    "Answer YES if the headline reports at least ONE of these:\n"
    "  A) A supply disruption — explosion, fire, outage, strike, closure, spill\n"
    "  B) A production or quota change — OPEC decision, field opening/shutdown, licence grant\n"
    "  C) A shipping route or rate event — rates at X-month high/low, route closure, canal blockage, rerouting\n"
    "  D) A new sanction, embargo, or export ban affecting oil, gas, or shipping\n"
    "  E) A physical infrastructure event — pipeline rupture, port blockage, terminal fire\n\n"
    "Answer NO if the headline is ONLY about:\n"
    "  - Analyst opinions, forecasts, or 'stable'/'flat' price commentary\n"
    "  - Corporate governance (CEO changes, pay deals, mergers not affecting supply)\n"
    "  - Routine completed operations (ship docked, cargo delivered, maintenance finished)\n"
    "  - Conferences, awards, or industry association meetings\n\n"
    "Headline: '{headline}'\n\n"
    "Answer ONLY 'YES' or 'NO'. No explanation."
)

_REASONING_PROMPT = """\
### MACRO CONTEXT ({date})
{macro_context}

### KNOWLEDGE BASE
{knowledge}

### NEWS EVENT
"{headline}"

### RULES — FOLLOW EXACTLY
1. DO NOT pick any company or ticker directly named or implied in the headline.
   That is a 1st-order trade. You are hunting for the INDIRECT winner.
2. Trace the physical supply chain step by step:
   - Supply increase  → Who refines or ships that new volume?
   - Fleet sanctions  → Who gains the market share vacated by the removed vessels?
   - Route disruption → Who carries the re-routed cargo?
                        Longer route = more ton-miles = tighter VLCC supply = higher spot day rates.
3. Prefer the ticker with the HIGHEST direct exposure to the identified effect.
   A tanker company with 85% spot exposure beats a refiner for a day-rate event.
   A high-complexity refiner beats a diversified major for a feedstock-supply event.
4. Use ONLY tickers that appear in the KNOWLEDGE BASE above.

### WORKED EXAMPLE (study this before answering)
News: "US expands Chevron licence to drill in Venezuela"
  WRONG (1st-order): CVX — Chevron is directly named. Obvious. Not useful.
  CORRECT (2nd-order causal chain):
    More Venezuelan heavy crude produced
    → needs to move from Caribbean to US Gulf Coast
    → Teekay's Aframax/Suezmax fleet specialises in exactly this regional route
    → higher Aframax utilisation → higher day rates → TNK LONG
    Also valid: PBF — high coking capacity runs Venezuelan heavy sour at a margin
    advantage vs competitors, so cheaper feedstock widens its crack spread.

### YOUR ANALYSIS
Apply the same step-by-step causal-chain reasoning to the news event above.
Think: what changes physically? Who in the knowledge base captures that change
without being directly named in the headline?

Output ONLY in this exact format (no preamble, no extra text):
- Ticker: <symbol from KNOWLEDGE BASE>
- Signal: <LONG or SHORT>
- Confidence: <0-100>%
- Rationale: <Exactly 2 sentences. Sentence 1: the causal chain (what physically changes). Sentence 2: why THIS ticker captures it better than the alternatives.>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze(headline: str) -> Optional[Signal]:
    """Run the 3-stage pipeline on a single news headline.

    Returns a Signal if a trade opportunity is identified, or None if the
    headline is filtered out at Stage 1 or 2, or if Stage 3 returns a
    malformed response.
    """
    cfg = get_config()
    headline_lower = headline.lower()

    # ------------------------------------------------------------------
    # Stage 1: Keyword filter — zero cost, three sub-tiers
    # ------------------------------------------------------------------

    # Sub-tier A: hard-NO keywords — always noise, skip LLM entirely.
    # Checked FIRST so corporate governance / analyst commentary that also
    # contains "oil" or "refinery" is blocked before reaching Stage 2.
    if any(k in headline_lower for k in cfg.no_pass_keywords):
        logger.info("S1 BLOCK | Hard-no keyword matched  | %s", headline)
        return None

    # Sub-tier B + C: must contain at least one keyword to proceed at all.
    all_keywords = cfg.high_alpha_keywords + cfg.general_keywords
    if not any(k in headline_lower for k in all_keywords):
        logger.info("S1 SKIP  | No keywords matched      | %s", headline)
        return None

    # Sub-tier B: high-alpha VIP — deterministic pass, skip Stage 2 sieve.
    vip_pass = any(k in headline_lower for k in cfg.high_alpha_keywords)

    if vip_pass:
        logger.info("S1 VIP   | High-alpha keyword       | %s", headline)
    else:
        logger.info("S1 PASS  | General keyword matched  | %s", headline)

    # ------------------------------------------------------------------
    # Stage 2: Mini-LLM sieve — skipped if VIP Pass
    # ------------------------------------------------------------------
    if not vip_pass:
        sieve_prompt = _SIEVE_PROMPT.format(headline=headline)
        try:
            sieve_response = ollama.generate(
                prompt=sieve_prompt,
                model=cfg.sieve_model,
                timeout=cfg.sieve_timeout,
                keep_alive=0,
                temperature=0.0,  # Deterministic — the 1B model must be consistent
            )
            logger.debug("S2 sieve response: %s", sieve_response.strip()[:80])

            if sieve_says_no(sieve_response):
                logger.info("S2 SKIP  | Sieve: not trade-relevant | %s", headline)
                return None

            logger.info("S2 PASS  | Sieve: trade-relevant      | %s", headline)

        except OllamaError as exc:
            # Sieve failure is non-fatal: a missed signal is worse than a
            # spurious Stage 3 call. Proceed with a warning.
            logger.warning("S2 WARN  | Sieve error (proceeding to S3): %s", exc)

    # ------------------------------------------------------------------
    # Stage 3: DeepSeek-R1-8B reasoning engine
    # ------------------------------------------------------------------
    logger.info("S3 START | Triggering reasoning engine | %s", headline)

    try:
        with cfg.knowledge_base_path.open("r", encoding="utf-8") as f:
            knowledge = json.load(f)
    except FileNotFoundError:
        logger.error("Knowledge base not found at %s", cfg.knowledge_base_path)
        return None

    macro_context = knowledge.pop("macro_context", {})

    reasoning_prompt = _REASONING_PROMPT.format(
        date=datetime.now().strftime("%B %Y"),
        macro_context=json.dumps(macro_context, indent=2),
        knowledge=json.dumps(knowledge, indent=2),
        headline=headline,
    )

    try:
        raw_response = ollama.generate(
            prompt=reasoning_prompt,
            model=cfg.reasoning_model,
            timeout=cfg.reasoning_timeout,
            keep_alive=0,
        )
    except OllamaError as exc:
        logger.error("S3 ERROR | Reasoning engine failed: %s", exc)
        return None

    signal = parse_signal(raw_response, headline)
    if signal is None:
        return None

    log_signal(signal)
    logger.info(
        "S3 SIGNAL| %s %s @ %d%% | %s",
        signal.ticker,
        signal.direction,
        signal.confidence,
        headline,
    )
    return signal
