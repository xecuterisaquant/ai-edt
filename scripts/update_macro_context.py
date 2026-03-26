"""Tier 1 KB maintenance — auto-refresh data/macro_context.json.

Reads recent headlines and signals from the SQLite database, then asks
Gemini to update the macro_context situations list to reflect the current
market regime.  A backup is written before any change is applied.

Usage:
    python -m scripts.update_macro_context               # default: last 7 days
    python -m scripts.update_macro_context --days 14
    python -m scripts.update_macro_context --dry-run     # print proposed JSON, don't write
    python -m scripts.update_macro_context --force       # skip confirmation prompt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_UPDATE_PROMPT = """\
You are a macro market analyst maintaining a structured knowledge base for an \
oil & gas event-driven trading system.

Today's date: {today}

CURRENT MACRO CONTEXT (JSON):
{current_macro}

RECENT HEADLINES PROCESSED BY THE SYSTEM (last {days} days, {n_headlines} total):
{headlines}

RECENT TRADE SIGNALS GENERATED (last {days} days):
{signals}

TASK:
Update the macro_context JSON to accurately reflect the current market regime.

Rules:
1. Keep existing situations that are still active and relevant today.
2. Update the "detail" field if the situation has evolved materially.
3. Add NEW situations for themes that clearly emerged in the recent headlines.
4. REMOVE situations that are stale — no relevant headlines in the period and \
   no longer affecting physical cargo flows or shipping rates.
5. Update "date" to today's month and year.
6. Each situation MUST have exactly three fields: "theme", "detail", \
   "physical_effect". Do not add or rename fields.
7. "physical_effect" must be a concise, mechanistic description of the \
   1st-order physical impact on crude/product flows or vessel supply/demand. \
   No vague sentiment — state the barrel or berth-day effect directly.

Return ONLY a valid JSON object with this exact structure — no markdown, no \
explanation, no code fences:
{{
  "date": "Month YYYY",
  "situations": [
    {{
      "theme": "...",
      "detail": "...",
      "physical_effect": "..."
    }}
  ]
}}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_recent_headlines(db_path: Path, days: int) -> list[str]:
    """Return distinct headline link entries from the last *days* days.

    We use headlines_seen as a proxy for what the watcher actually processed.
    The link field contains the article URL — we show a count and sample, not
    every URL.
    """
    import sqlite3

    if not db_path.exists():
        return []

    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT link, seen_at, feed_source FROM headlines_seen WHERE seen_at >= ? ORDER BY seen_at DESC",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()
    return [f"[{r[2] or 'unknown'}] {r[0]}" for r in rows]


def _load_recent_signals(db_path: Path, days: int) -> list[dict]:
    """Return recent signals as plain dicts."""
    import sqlite3

    if not db_path.exists():
        return []

    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT ticker, direction, confidence, headline, created_utc
               FROM signals WHERE created_utc >= ?
               ORDER BY created_utc DESC""",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _call_gemini(prompt: str, model: str, timeout: int) -> str:
    """Call Gemini directly without importing the full ai_edt package."""
    import os

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment / .env")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            http_options=types.HttpOptions(timeout=timeout * 1000),
        ),
    )
    return response.text or ""


def _validate_macro(data: object) -> None:
    """Raise ValueError if *data* does not match the expected macro schema."""
    if not isinstance(data, dict):
        raise ValueError("Top-level value must be a JSON object")
    if "date" not in data:
        raise ValueError("Missing required key 'date'")
    situations = data.get("situations")
    if not isinstance(situations, list):
        raise ValueError("'situations' must be a list")
    for i, s in enumerate(situations):
        for field in ("theme", "detail", "physical_effect"):
            if field not in s:
                raise ValueError(f"situations[{i}] missing field '{field}'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auto-refresh data/macro_context.json via LLM.")
    parser.add_argument("--days", type=int, default=7, help="Look-back window in days (default: 7)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print proposed JSON without writing"
    )
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model to use")
    parser.add_argument("--timeout", type=int, default=120, help="API timeout in seconds")
    parser.add_argument(
        "--macro-path",
        default=str(PROJECT_ROOT / "data" / "macro_context.json"),
        help="Path to macro_context.json",
    )
    parser.add_argument(
        "--db-path",
        default=str(PROJECT_ROOT / "signals" / "signals.db"),
        help="Path to signals.db",
    )
    args = parser.parse_args(argv)

    macro_path = Path(args.macro_path)
    db_path = Path(args.db_path)

    # Load current macro context
    if not macro_path.exists():
        print(f"ERROR: macro context file not found at {macro_path}", file=sys.stderr)
        return 1

    current_macro = json.loads(macro_path.read_text(encoding="utf-8"))

    # Load recent data from DB
    print(f"Loading headlines and signals from last {args.days} days...")
    headlines = _load_recent_headlines(db_path, args.days)
    signals = _load_recent_signals(db_path, args.days)

    print(f"  {len(headlines)} headlines processed, {len(signals)} signals generated")

    # Summarise signals for the prompt (avoid enormous context)
    signal_summary = []
    for s in signals[:50]:  # cap at 50 most recent
        signal_summary.append(
            f"  {s['created_utc'][:10]} | {s['ticker']} {s['direction']} "
            f"@ {s['confidence']}% | {s['headline']}"
        )

    # Sample headlines (cap at 200 for context length)
    headline_sample = headlines[:200]
    if len(headlines) > 200:
        print(f"  (showing 200 of {len(headlines)} headlines in prompt)")

    today = datetime.now().strftime("%B %Y")
    prompt = _UPDATE_PROMPT.format(
        today=today,
        current_macro=json.dumps(current_macro, indent=2),
        days=args.days,
        n_headlines=len(headlines),
        headlines="\n".join(headline_sample) if headline_sample else "  (none — db may be empty)",
        signals="\n".join(signal_summary) if signal_summary else "  (none)",
    )

    print(f"\nCalling Gemini ({args.model}) to update macro context...")
    try:
        raw = _call_gemini(prompt, model=args.model, timeout=args.timeout)
    except Exception as exc:
        print(f"ERROR calling Gemini: {exc}", file=sys.stderr)
        return 1

    # Strip markdown code fences if Gemini wrapped the output
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    # Parse and validate
    try:
        proposed = json.loads(raw)
        _validate_macro(proposed)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"\nERROR: LLM output failed validation: {exc}", file=sys.stderr)
        print("\nRaw output:\n", raw, file=sys.stderr)
        return 1

    # Show diff summary
    old_themes = {s["theme"] for s in current_macro.get("situations", [])}
    new_themes = {s["theme"] for s in proposed.get("situations", [])}
    added = new_themes - old_themes
    removed = old_themes - new_themes
    kept = old_themes & new_themes

    print(f"\n--- Proposed macro_context ({proposed.get('date', '?')}) ---")
    if added:
        print(f"  + ADDED   ({len(added)}): {', '.join(sorted(added))}")
    if removed:
        print(f"  - REMOVED ({len(removed)}): {', '.join(sorted(removed))}")
    if kept:
        print(f"  ~ KEPT    ({len(kept)}): {', '.join(sorted(kept))}")

    if args.dry_run:
        print("\n--- Proposed JSON (dry-run, not written) ---")
        print(json.dumps(proposed, indent=2, ensure_ascii=False))
        return 0

    # Confirmation
    if not args.force:
        answer = input("\nApply this update? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted — no changes written.")
            return 0

    # Backup and write
    backup_path = macro_path.with_suffix(".json.bak")
    shutil.copy2(macro_path, backup_path)
    print(f"Backup written to {backup_path}")

    macro_path.write_text(
        json.dumps(proposed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Updated {macro_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
