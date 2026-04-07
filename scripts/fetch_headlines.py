"""Fetch historical headlines from GDELT Doc 2.0 API for backtesting.

Queries GDELT's free article-list endpoint and writes a CSV compatible
with ``scripts/backtest.py``.

GDELT docs:  https://blog.gdeltproject.org/gdelt-doc-2-0-api-exploring-the-world/

Usage
-----
    python -m scripts.fetch_headlines -o headlines.csv
    python -m scripts.fetch_headlines --from 2025-06-01 --to 2026-03-31
    python -m scripts.fetch_headlines --query "OPEC oil sanctions tanker" --limit 500
    python -m scripts.fetch_headlines -o headlines.csv --domain reuters.com --limit 250
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# GDELT Doc 2.0 API configuration
# ---------------------------------------------------------------------------

_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Maximum records per single API call (GDELT caps at 250).
_GDELT_MAX_PER_CALL = 250

# Default keyword groups that align with the pipeline's Stage 1 sector keywords.
# Broad enough to capture relevant headlines, narrow enough to avoid noise.
_DEFAULT_QUERY_GROUPS = [
    # Upstream / geopolitical
    "oil sanctions OR crude sanctions OR Venezuela oil OR Iran oil OR OPEC production",
    # Shipping / logistics
    "tanker attack OR Red Sea shipping OR Suez Canal OR Hormuz strait OR VLCC rates",
    # Refinery / downstream
    "refinery outage OR refinery fire OR crack spread OR gasoline inventory",
    # LNG / natural gas
    "LNG terminal OR natural gas pipeline OR LNG export OR gas supply disruption",
    # Broader energy
    "oil pipeline explosion OR bpd cut OR crude inventory OR strategic petroleum reserve",
]

# Polite delay between GDELT API calls to avoid rate-limiting.
_REQUEST_DELAY_S = 1.5

# HTTP timeout for each GDELT request.
_HTTP_TIMEOUT_S = 30


# ---------------------------------------------------------------------------
# API interaction
# ---------------------------------------------------------------------------


def _gdelt_fetch(
    query: str,
    *,
    start: str,
    end: str,
    max_records: int = _GDELT_MAX_PER_CALL,
    domain: str | None = None,
    sort: str = "DateDesc",
) -> list[dict]:
    """Call the GDELT Doc 2.0 artlist endpoint and return parsed articles.

    Each dict has keys: ``url``, ``title``, ``seendate``, ``domain``.
    Returns an empty list on HTTP or parse errors.
    """
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": min(max_records, _GDELT_MAX_PER_CALL),
        "startdatetime": start,
        "enddatetime": end,
        "format": "json",
        "sort": sort,
    }
    if domain:
        params["query"] += f" domain:{domain}"

    try:
        resp = requests.get(_API_URL, params=params, timeout=_HTTP_TIMEOUT_S)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  GDELT request failed: {exc}", file=sys.stderr)
        return []

    try:
        data = resp.json()
    except ValueError:
        print("  GDELT returned non-JSON response.", file=sys.stderr)
        return []

    return data.get("articles", [])


def _parse_gdelt_date(seendate: str) -> str | None:
    """Convert GDELT's ``YYYYMMDDTHHMMSSZ`` format to ISO-8601.

    Returns None for unparseable values.
    """
    try:
        dt = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        return dt.isoformat()
    except (ValueError, TypeError):
        return None


def fetch_headlines(
    *,
    queries: list[str] | None = None,
    start_date: str,
    end_date: str,
    limit: int = 500,
    domain: str | None = None,
) -> list[dict]:
    """Fetch headlines from GDELT across multiple query groups.

    Returns de-duplicated dicts with keys: ``datetime``, ``headline``,
    ``feed_source``.
    """
    if queries is None:
        queries = _DEFAULT_QUERY_GROUPS

    # Convert YYYY-MM-DD → YYYYMMDDHHMMSS for the GDELT API.
    gdelt_start = start_date.replace("-", "") + "000000"
    gdelt_end = end_date.replace("-", "") + "235959"

    seen_titles: set[str] = set()
    results: list[dict] = []
    per_query_limit = max(limit // len(queries), 50)

    for i, query in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] Querying: {query[:60]}...")
        articles = _gdelt_fetch(
            query,
            start=gdelt_start,
            end=gdelt_end,
            max_records=min(per_query_limit, _GDELT_MAX_PER_CALL),
            domain=domain,
        )
        for art in articles:
            title = (art.get("title") or "").strip()
            if not title or title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())
            dt_iso = _parse_gdelt_date(art.get("seendate", ""))
            if dt_iso is None:
                continue
            results.append({
                "datetime": dt_iso,
                "headline": title,
                "feed_source": art.get("domain", "gdelt"),
            })

        if len(results) >= limit:
            break
        if i < len(queries):
            time.sleep(_REQUEST_DELAY_S)

    # Sort chronologically and trim to limit.
    results.sort(key=lambda r: r["datetime"])
    return results[:limit]


def write_csv(rows: list[dict], path: Path) -> int:
    """Write headline rows to a CSV file. Returns the number of rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["datetime", "headline", "feed_source"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    nine_months_ago = (datetime.now(UTC) - timedelta(days=270)).strftime("%Y-%m-%d")
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    p = argparse.ArgumentParser(
        prog="python -m scripts.fetch_headlines",
        description="Fetch historical headlines from GDELT for backtesting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("data/headlines.csv"),
        metavar="PATH",
        help="Output CSV path.",
    )
    p.add_argument(
        "--from",
        type=str,
        default=nine_months_ago,
        dest="start_date",
        metavar="YYYY-MM-DD",
        help="Start date for headline search.",
    )
    p.add_argument(
        "--to",
        type=str,
        default=today,
        dest="end_date",
        metavar="YYYY-MM-DD",
        help="End date for headline search.",
    )
    p.add_argument(
        "--query",
        type=str,
        default=None,
        metavar="TERMS",
        help="Custom GDELT query (overrides default query groups).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum total headlines to fetch.",
    )
    p.add_argument(
        "--domain",
        type=str,
        default=None,
        metavar="DOMAIN",
        help="Restrict results to a specific domain (e.g., reuters.com).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    print(f"Fetching headlines from GDELT ({args.start_date} → {args.end_date})...")
    print(f"  Limit: {args.limit} | Domain: {args.domain or 'any'}")

    queries = [args.query] if args.query else None

    rows = fetch_headlines(
        queries=queries,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
        domain=args.domain,
    )

    if not rows:
        print("\nNo headlines returned from GDELT. Try broader query terms or date range.")
        return 1

    n = write_csv(rows, args.output)
    print(f"\nWrote {n} headlines to {args.output}")
    print(f"  Date range: {rows[0]['datetime'][:10]} → {rows[-1]['datetime'][:10]}")
    print(f"\nNext step: python -m scripts.backtest --csv {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
