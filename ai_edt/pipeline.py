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

from ai_edt import gemini as _gemini
from ai_edt import ollama
from ai_edt.config import get_config
from ai_edt.gemini import GeminiError
from ai_edt.logger import get_logger
from ai_edt.ollama import OllamaError
from ai_edt.signals import Signal, log_signal, parse_signal, sieve_says_no

logger = get_logger("pipeline")


# ---------------------------------------------------------------------------
# LLM dispatch — single provider-routing point for both S2 and S3
# ---------------------------------------------------------------------------


def _call_llm(
    prompt: str,
    model_gemini: str,
    model_ollama: str,
    timeout: int,
    *,
    keep_alive: int = 0,
    temperature: float | None = None,
) -> str:
    """Route an LLM call to the configured provider.

    Abstracts the gemini-vs-ollama branching so callers only need one
    call site instead of duplicated if/else blocks.
    """
    cfg = get_config()
    if cfg.reasoning_provider == "gemini":
        return _gemini.generate(prompt=prompt, model=model_gemini, timeout=timeout)
    return ollama.generate(
        prompt=prompt,
        model=model_ollama,
        timeout=timeout,
        keep_alive=keep_alive,
        temperature=temperature,
    )


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
   That is a 1st-order trade. You are hunting for the INDIRECT winner OR loser.
2. First decide direction — LONG or SHORT — by tracing the causal chain:

   LONG (2nd-order beneficiary):
   - Supply disruption    → competing refiners gain as the disrupted refinery's output
                            is removed, widening crack spreads industry-wide.
   - Supply increase      → who refines or ships that new volume?
   - Fleet sanctions      → who captures the market share vacated by the removed vessels?
   - Route disruption     → who carries the re-routed cargo?
                            Longer route = more ton-miles = tighter VLCC supply = higher spot day rates.

   SHORT (2nd-order loser):
   - VLCC oversupply      → which tanker operator has the HIGHEST spot exposure to a
                            day-rate collapse? High spot exposure cuts both ways.
   - Crude price spike    → which refiner has the thinnest margins / lowest complexity
                            to absorb the higher feedstock cost?
   - Supply glut          → which refiner loses its feedstock cost advantage when
                            prices normalise?

3. Prefer the ticker with the HIGHEST direct exposure to the identified effect.
   A high-complexity refiner wins a heavy-crude feedstock event.
   A tanker with 85% spot exposure wins AND loses hardest on rate events.
4. Use ONLY tickers that appear in the KNOWLEDGE BASE above.
5. If the effect is genuinely ambiguous, commit to the most probable direction.

### WORKED EXAMPLE — LONG
News: "US expands Chevron licence to drill in Venezuela"
  WRONG (1st-order): CVX — Chevron is directly named. Obvious. Not useful.
  CORRECT (2nd-order LONG):
    More Venezuelan heavy crude produced
    → needs to move from Caribbean to US Gulf Coast
    → Teekay's Aframax/Suezmax fleet specialises in exactly this regional route
    → higher Aframax utilisation → higher day rates → TNK LONG
    Also valid: PBF — cheaper heavy sour feedstock widens crack spread vs competitors.

### WORKED EXAMPLE — SHORT
News: "Global VLCC newbuild deliveries surge — 5% fleet growth this quarter"
  WRONG (1st-order): any shipyard — not in the knowledge base.
  CORRECT (2nd-order SHORT):
    More VLCCs enter service → tanker supply increases relative to cargo demand
    → charterers gain rate-setting power → VLCC spot day rates fall
    → FRO's 85% spot exposure means earnings decline almost immediately → FRO SHORT

### YOUR ANALYSIS
Apply the same step-by-step causal-chain reasoning to the news event above.
Think: what changes physically? Is it bullish or bearish for 2nd-order players?
Pick the ONE ticker in the KNOWLEDGE BASE that captures it most directly.

Output ONLY in this exact format (no preamble, no extra text):
- Ticker: <symbol from KNOWLEDGE BASE>
- Signal: <LONG or SHORT>
- Confidence: <0-100>%
- Rationale: <Exactly 2 sentences. Sentence 1: the causal chain (what physically changes and in which direction). Sentence 2: why THIS ticker captures it better than the alternatives.>
"""

# ---------------------------------------------------------------------------
# Sector-routing hints for Stage 3 KB filtering
#
# _select_relevant_kb() uses these to trim the knowledge base before building
# the reasoning prompt. On a sector-specific headline this cuts token count
# ~40-50%, directly reducing inference time on memory-constrained hardware.
#
# Design principle: put a term in ONE set only if it is unambiguously that
# sector. Leave cross-sector terms (venezuela, opec, bpd, explosion) out of
# both sets — they fall through to the full KB, which is the safe default.
# ---------------------------------------------------------------------------

_LOGISTICS_HINTS = frozenset(
    {
        "vlcc",
        "aframax",
        "suezmax",
        "tanker rate",
        "day rate",
        "freight rate",
        "spot rate",
        "shipping rate",
        "rerouting",
        "cape of good hope",
        "hormuz",
        "malacca",
        "red sea",
        "houthi",
        "iran",
        "yemen",
        "shadow fleet",
        "strait",
        "canal",
    }
)

_REFINERY_HINTS = frozenset(
    {
        "refinery",
        "refining",
        "crack spread",
        "crack",
        "feedstock",
        "heavy sour",
        "distillate",
        "gasoline",
        "diesel",
    }
)

# Named pipeline systems — unambiguously midstream, not shipping or refining.
_PIPELINE_HINTS = frozenset(
    {
        "enbridge",
        "keystone",
        "druzhba",
        "espo",
    }
)


def _select_relevant_kb(headline_lower: str, knowledge: dict) -> dict:
    """Return the KB subset most relevant to this headline's sector.

    Returns the full KB when a headline matches multiple sectors or none,
    ensuring no signal is missed due to over-filtering.
    Upstream context is always included because producers underlie all
    supply-chain events.
    """
    has_shipping = any(h in headline_lower for h in _LOGISTICS_HINTS)
    has_refinery = any(h in headline_lower for h in _REFINERY_HINTS)
    has_pipeline = any(h in headline_lower for h in _PIPELINE_HINTS)

    sector_hits = sum([has_shipping, has_refinery, has_pipeline])
    if sector_hits != 1:
        return knowledge  # ambiguous / cross-sector / none → full KB

    # Single clear sector: keep that section + upstream (always context-relevant).
    keep: set[str] = {"upstream_data"}
    if has_shipping:
        keep.add("shipping_data")
        logger.debug("S3 KB    | Sector filter: shipping")
    elif has_refinery:
        keep.add("refinery_data")
        logger.debug("S3 KB    | Sector filter: refinery")
    else:
        keep.add("midstream_data")
        logger.debug("S3 KB    | Sector filter: pipeline")

    return {k: v for k, v in knowledge.items() if k in keep}


# ---------------------------------------------------------------------------
# Shared keyword pre-filter (used by pipeline + watcher)
# ---------------------------------------------------------------------------


def stage1_matches(headline_lower: str) -> str | None:
    """Apply the Stage 1 keyword filter and return the match tier.

    Returns:
        ``"no_pass"`` - headline blocked by a no-pass keyword.
        ``"vip"``     - headline matched a high-alpha keyword (skip sieve).
        ``"general"`` - headline matched a general keyword (proceed to sieve).
        ``None``      - no keywords matched at all.
    """
    cfg = get_config()
    if any(k in headline_lower for k in cfg.no_pass_keywords):
        return "no_pass"
    if any(k in headline_lower for k in cfg.high_alpha_keywords):
        return "vip"
    if any(k in headline_lower for k in cfg.general_keywords):
        return "general"
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze(headline: str, feed_source: str = "") -> Signal | None:
    """Run the 3-stage pipeline on a single news headline.

    *feed_source* identifies the RSS feed or data source that produced this
    headline.  It is stored with the signal for feed-quality analysis.

    Returns a Signal if a trade opportunity is identified, or None if the
    headline is filtered out at Stage 1 or 2, or if Stage 3 returns a
    malformed response.
    """
    cfg = get_config()
    headline_lower = headline.lower()

    # ------------------------------------------------------------------
    # Stage 1: Keyword filter — zero cost, three sub-tiers
    # ------------------------------------------------------------------

    tier = stage1_matches(headline_lower)

    if tier == "no_pass":
        logger.info("S1 BLOCK | Hard-no keyword matched  | %s", headline)
        return None
    if tier is None:
        logger.info("S1 SKIP  | No keywords matched      | %s", headline)
        return None

    vip_pass = tier == "vip"

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
            sieve_response = _call_llm(
                prompt=sieve_prompt,
                model_gemini=cfg.gemini_model,
                model_ollama=cfg.sieve_model,
                timeout=cfg.sieve_timeout,
                keep_alive=0,
                temperature=0.0,
            )
            logger.debug("S2 sieve response: %s", sieve_response.strip()[:80])

            if sieve_says_no(sieve_response):
                logger.info("S2 SKIP  | Sieve: not trade-relevant | %s", headline)
                return None

            logger.info("S2 PASS  | Sieve: trade-relevant      | %s", headline)

        except (OllamaError, GeminiError) as exc:
            # Sieve failure is non-fatal: a missed signal is worse than a
            # spurious Stage 3 call. Proceed with a warning.
            logger.warning("S2 WARN  | Sieve error (proceeding to S3): %s", exc)

    # ------------------------------------------------------------------
    # Stage 3: DeepSeek-R1-8B reasoning engine
    # ------------------------------------------------------------------
    logger.info("S3 START | Triggering reasoning engine | %s", headline)

    try:
        with cfg.macro_context_path.open("r", encoding="utf-8") as f:
            macro_context = json.load(f)
    except FileNotFoundError:
        logger.warning(
            "Macro context not found at %s — using empty context", cfg.macro_context_path
        )
        macro_context = {}

    try:
        with cfg.knowledge_base_path.open("r", encoding="utf-8") as f:
            knowledge = json.load(f)
    except FileNotFoundError:
        logger.error("Knowledge base not found at %s", cfg.knowledge_base_path)
        return None

    filtered_knowledge = _select_relevant_kb(headline_lower, knowledge)

    reasoning_prompt = _REASONING_PROMPT.format(
        date=datetime.now().strftime("%B %Y"),
        macro_context=json.dumps(macro_context, indent=2),
        knowledge=json.dumps(filtered_knowledge, indent=2),
        headline=headline,
    )

    try:
        logger.debug(
            "S3 provider | %s | model=%s",
            cfg.reasoning_provider,
            cfg.gemini_model if cfg.reasoning_provider == "gemini" else cfg.reasoning_model,
        )
        raw_response = _call_llm(
            prompt=reasoning_prompt,
            model_gemini=cfg.gemini_model,
            model_ollama=cfg.reasoning_model,
            timeout=cfg.reasoning_timeout,
            keep_alive=0,
        )
    except (OllamaError, GeminiError) as exc:
        logger.error("S3 ERROR | Reasoning engine failed: %s", exc)
        return None

    signal = parse_signal(raw_response, headline)
    if signal is None:
        return None

    signal.feed_source = feed_source
    log_signal(signal)
    logger.info(
        "S3 SIGNAL| %s %s @ %d%% | %s",
        signal.ticker,
        signal.direction,
        signal.confidence,
        headline,
    )
    return signal
