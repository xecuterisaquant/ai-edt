"""Tier 2 KB maintenance — generate a new KB entry from source URLs.

Fetches one or more URLs, strips HTML, and asks Gemini to produce a
structured KB entry for a given ticker, following the exact schema used in
data/market_knowledge.json.

The output is printed to stdout — inspect it, then paste the JSON object
into the appropriate section of market_knowledge.json.

Usage:
    python -m scripts.research_kb_entry --ticker STNG --section shipping_data \\
        --urls https://example.com/stng-profile https://example.com/stng-fleet

    python -m scripts.research_kb_entry --ticker MPC --section refinery_data \\
        --urls https://example.com/mpc-overview

Options:
    --ticker TICKER           Ticker symbol (e.g. FRO, PBF, TNK).
    --section SECTION         KB section to place the entry in.
                              One of: shipping_data, refinery_data,
                              upstream_data, midstream_data.
    --urls URL [URL ...]      One or more source URLs to fetch and analyse.
    --model MODEL             Gemini model (default: gemini-2.5-flash).
    --timeout SECS            API timeout in seconds (default: 120).
    --kb-path PATH            Path to market_knowledge.json for schema context.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Section schemas — the LLM is shown an existing entry as a formatting example
# ---------------------------------------------------------------------------

# Map each section to the ticker whose entry we use as the in-prompt template.
_SECTION_TEMPLATES: dict[str, str] = {
    "shipping_data": "FRO",
    "refinery_data": "PBF",
    "upstream_data": "CVX",
    "midstream_data": "ENB",
}

_VALID_SECTIONS = frozenset(_SECTION_TEMPLATES.keys())

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_RESEARCH_PROMPT = """\
You are building a structured knowledge base for an oil & gas event-driven \
trading system that analyses news headlines and identifies second-order trade \
opportunities.

TARGET TICKER: {ticker}
TARGET SECTION: {section}

EXAMPLE ENTRY (use this EXACT field structure for your output):
{template_entry}

SOURCE MATERIAL (fetched from {n_urls} URL(s)):
---
{source_text}
---

TASK:
Write a new KB entry for {ticker} following the exact field structure shown \
in the example above.

Rules:
1. Copy the field names exactly from the example — do NOT add, rename, or \
   remove fields.
2. "primary_sensitivity" must explain the precise mechanism by which a news \
   event translates into an earnings impact. State the causal chain: \
   event → physical effect → rate/margin/price change → earnings change.
3. "constraints" must document what this company does NOT benefit from — \
   edge cases where a naïve analysis would pick the wrong ticker. Be \
   specific: name the competing ticker and the scenario where it is \
   preferred.
4. Write for a quantitative analyst: precise, mechanistic, no vague \
   sentiment. Use exact figures (fleet size, market share, route distances) \
   where available in the source material.
5. Return ONLY the JSON object for this one entry — no markdown, no \
   explanation, no code fences. The output must be directly pasteable into \
   the "{section}" array in market_knowledge.json.
"""

# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

# Tags that indicate navigation/boilerplate — strip their content entirely.
_BLOCK_TAGS_RE = re.compile(
    r"<(script|style|nav|header|footer|aside|noscript)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _strip_html(html: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = _BLOCK_TAGS_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", text)
    # Decode common HTML entities
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .replace("&#39;", "'")
        .replace("&quot;", '"')
    )
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _fetch_url(url: str, timeout: int = 15) -> str:
    """Fetch a URL and return stripped plain text."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; AI-EDT-KB-Research/1.0; "
                "+https://github.com/xecuterisaquant/ai-edt)"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        charset = "utf-8"
        content_type = resp.headers.get("Content-Type", "")
        if "charset=" in content_type:
            charset = content_type.split("charset=")[-1].strip()
        raw = resp.read().decode(charset, errors="replace")
    return _strip_html(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_template_entry(kb_path: Path, section: str, ticker: str) -> dict[str, Any] | None:
    """Find the template ticker's entry in the KB for use as a schema example."""
    if not kb_path.exists():
        return None
    kb = json.loads(kb_path.read_text(encoding="utf-8"))
    for entry in kb.get(section, []):
        if entry.get("ticker") == ticker:
            return entry
    return None


def _call_gemini(prompt: str, model: str, timeout: int) -> str:
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a KB entry for a ticker from source URLs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. STNG")
    parser.add_argument(
        "--section",
        required=True,
        choices=sorted(_VALID_SECTIONS),
        help="Which KB section the entry belongs to",
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        required=True,
        metavar="URL",
        help="One or more source URLs to fetch",
    )
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model to use")
    parser.add_argument("--timeout", type=int, default=120, help="API timeout in seconds")
    parser.add_argument(
        "--kb-path",
        default=str(PROJECT_ROOT / "data" / "market_knowledge.json"),
        help="Path to market_knowledge.json (used to load template entry)",
    )
    args = parser.parse_args(argv)

    ticker = args.ticker.upper()
    kb_path = Path(args.kb_path)

    # Load template entry for schema context
    template_ticker = _SECTION_TEMPLATES[args.section]
    template_entry = _load_template_entry(kb_path, args.section, template_ticker)
    if template_entry is None:
        print(
            f"WARNING: could not find template entry for {template_ticker} in "
            f"{args.section} — LLM will receive no schema example.",
            file=sys.stderr,
        )
        template_json = "{}"
    else:
        template_json = json.dumps(template_entry, indent=2)

    # Fetch source URLs
    source_chunks: list[str] = []
    for url in args.urls:
        print(f"Fetching: {url}", file=sys.stderr)
        try:
            text = _fetch_url(url)
            # Cap each source at 8 000 chars to stay within context limits
            if len(text) > 8000:
                text = text[:8000] + "\n[...truncated...]"
            source_chunks.append(f"SOURCE: {url}\n{text}")
        except Exception as exc:
            print(f"  WARNING: could not fetch {url}: {exc}", file=sys.stderr)
            source_chunks.append(f"SOURCE: {url}\n[fetch failed: {exc}]")

    if not source_chunks:
        print("ERROR: no source content retrieved — aborting.", file=sys.stderr)
        return 1

    source_text = "\n\n---\n\n".join(source_chunks)

    prompt = _RESEARCH_PROMPT.format(
        ticker=ticker,
        section=args.section,
        template_entry=template_json,
        n_urls=len(args.urls),
        source_text=source_text,
    )

    print(f"\nCalling Gemini ({args.model}) to generate KB entry for {ticker}...", file=sys.stderr)
    try:
        raw = _call_gemini(prompt, model=args.model, timeout=args.timeout)
    except Exception as exc:
        print(f"ERROR calling Gemini: {exc}", file=sys.stderr)
        return 1

    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    # Validate it's parseable JSON
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object")
        if "ticker" not in parsed:
            raise ValueError("Entry has no 'ticker' field")
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"\nWARNING: output may not be valid JSON: {exc}", file=sys.stderr)
        print("\nRaw output:\n", raw)
        return 1

    # Pretty-print to stdout for copy-paste
    print(
        f"\n# ---- Copy the JSON below into the '{args.section}' array "
        f"in data/market_knowledge.json ----"
    )
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    print(
        f"\n# ---- End of entry for {ticker} ----\n"
        f"# Review carefully before committing. Key checks:\n"
        f"#   1. primary_sensitivity: is the causal chain precise?\n"
        f"#   2. constraints: does it call out the right competing tickers?\n"
        f"#   3. All facts (fleet size, routes, market share) verified?"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
