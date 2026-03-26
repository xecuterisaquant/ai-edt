"""Phase 3: Outcome Tracking — batch price backfill for open signals.

Scans the signals DB for rows with missing outcome prices and fills them
using yfinance 1h bars.  Designed to be run periodically — for example,
every hour via Windows Task Scheduler or a cron job.

Usage
-----
    python -m scripts.fetch_outcomes               # fill all pending signals
    python -m scripts.fetch_outcomes --dry-run     # preview without writing
    python -m scripts.fetch_outcomes --id 42       # fill one specific signal
    python -m scripts.fetch_outcomes --limit 100   # cap batch size
    python -m scripts.fetch_outcomes --max-age-days 30  # look back further
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_edt import db as _db
from ai_edt.logger import get_logger
from ai_edt.outcomes import fetch_prices_for_signal

logger = get_logger("fetch_outcomes")

_TICKER_W = 6  # column width for aligned output
_DIR_W = 5


def _process(row: dict, *, dry_run: bool) -> tuple[str, dict]:
    """Fetch prices for *row* and optionally persist them.

    Returns ``(status, prices)`` where status is one of:
        "filled"            — price_at_signal retrieved, DB updated (or would be)
        "signal_too_recent" — < 30 min old, skipped
        "no_price_data"     — yfinance returned empty DataFrame
        "no_price_at_signal"— nearest bar too far from signal time
        "yfinance_error:…"  — exception from yfinance
    """
    prices = fetch_prices_for_signal(
        ticker=row["ticker"],
        direction=row["direction"],
        created_utc=row["created_utc"],
    )
    note = prices.get("outcome_note")
    p0 = prices.get("price_at_signal")

    if p0 is not None:
        if not dry_run:
            _db.update_signal_prices(row["id"], prices)
        return "filled", prices

    return (note or "no_price_at_signal"), prices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill outcome prices for AI-EDT signals (Phase 3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview; do not write to DB.")
    parser.add_argument("--id", type=int, dest="signal_id", help="Process a single signal by ID.")
    parser.add_argument("--limit", type=int, default=200, help="Max signals per run.")
    parser.add_argument(
        "--min-age",
        type=int,
        default=30,
        dest="min_age_minutes",
        help="Min signal age in minutes before fetching prices.",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=7,
        dest="max_age_days",
        help="Max signal age in days — older signals are excluded.",
    )
    args = parser.parse_args(argv)

    if args.signal_id:
        row = _db.get_signal_by_id(args.signal_id)
        if row is None:
            print(f"Signal {args.signal_id} not found.", file=sys.stderr)
            return 1
        signals = [row]
    else:
        signals = _db.get_signals_needing_outcomes(
            min_age_minutes=args.min_age_minutes,
            max_age_days=args.max_age_days,
        )[: args.limit]

    if not signals:
        print("No signals pending outcome backfill.")
        return 0

    label = "  [dry-run]" if args.dry_run else ""
    print(f"Processing {len(signals)} signal(s){label}...")

    counts: dict[str, int] = {"filled": 0, "skipped": 0, "error": 0}

    for row in signals:
        try:
            status, prices = _process(row, dry_run=args.dry_run)
            ts = str(row["created_utc"])[:16]
            ticker = str(row["ticker"]).ljust(_TICKER_W)
            direction = str(row["direction"]).ljust(_DIR_W)
            p0 = prices.get("price_at_signal")
            p24 = prices.get("price_24h")
            pnl24 = prices.get("outcome_pnl_24h")

            if status == "filled":
                counts["filled"] += 1
                p0_str = f"  entry=${p0:.2f}" if p0 is not None else ""
                p24_str = f"  exit_24h=${p24:.2f}" if p24 is not None else ""
                pnl_str = f"  pnl_24h={pnl24:+.2%}" if pnl24 is not None else ""
                print(
                    f"  [{row['id']:>5}] {ticker} {direction}  {ts}  \u2713{p0_str}{p24_str}{pnl_str}"
                )
            elif status == "signal_too_recent":
                counts["skipped"] += 1
                print(f"  [{row['id']:>5}] {ticker} {direction}  {ts}  \u2013 {status}")
            else:
                counts["error"] += 1
                print(f"  [{row['id']:>5}] {ticker} {direction}  {ts}  \u2717 {status}")

        except Exception as exc:  # noqa: BLE001
            counts["error"] += 1
            logger.exception("Unexpected error for signal %s: %s", row["id"], exc)

    print(
        f"\nDone: {counts['filled']} filled,"
        f" {counts['skipped']} skipped,"
        f" {counts['error']} errors."
    )
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
