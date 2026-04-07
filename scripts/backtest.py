"""Phase 6: Backtesting Engine — historical headline replay + performance report.

Accepts a CSV of past headlines, runs each through the full 3-stage pipeline,
persists signals in the DB, fetches price outcomes, and prints a performance
summary.

CSV format (with header row)::

    datetime,headline,feed_source
    2026-03-10T14:32:00Z,Houthi attack on tanker Red Sea,reuters
    2026-03-11T09:15:00Z,OPEC+ confirms 1M bpd cut extension,bloomberg

``datetime`` must be ISO-8601 (UTC preferred; naive timestamps assumed UTC).
``feed_source`` is optional — omit the column or leave the cell blank.

Usage
-----
    python -m scripts.backtest --csv events.csv
    python -m scripts.backtest --csv events.csv --dry-run
    python -m scripts.backtest --csv events.csv --max-cost 0.50
    python -m scripts.backtest --csv events.csv --skip-outcomes
    python -m scripts.backtest --report                         # report only (no replay)
    python -m scripts.backtest --report --min-confidence 70

Options
-------
--csv PATH              CSV file of headlines to replay.
--dry-run               Run pipeline but do not persist signals or fetch prices.
--max-cost FLOAT        Abort replay if cumulative estimated cost exceeds this
                        value in USD (default: 2.00).
--skip-outcomes         Do not fetch price outcomes after replay.
--report                Print performance report from DB signals (no replay).
--min-confidence INT    Minimum confidence to include in performance stats (default: 0).
--ticker TICKER         Filter report to one ticker.
--since DAYS            Include only signals from the last N days in report (default: 90).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_edt import db as _db
from ai_edt import pipeline
from ai_edt.logger import get_logger
from ai_edt.outcomes import fetch_prices_for_signal

logger = get_logger("backtest")

# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


class _HeadlineRow(NamedTuple):
    dt: str          # ISO-8601 UTC timestamp string
    headline: str
    feed_source: str


def load_csv(path: Path) -> list[_HeadlineRow]:
    """Parse a headline CSV and return a list of _HeadlineRow.

    Required columns: ``datetime``, ``headline``.
    Optional column:  ``feed_source`` (defaults to ``"backtest"``).

    Rows with blank headlines are skipped.  Timestamps that cannot be
    parsed are skipped with a warning.
    """
    rows: list[_HeadlineRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {path} appears to be empty or has no header row.")
        header = [c.strip().lower() for c in reader.fieldnames]
        if "datetime" not in header or "headline" not in header:
            raise ValueError(
                f"CSV must have 'datetime' and 'headline' columns. Found: {reader.fieldnames}"
            )
        for i, raw in enumerate(reader, start=2):
            headline = raw.get("headline", "").strip()
            if not headline:
                continue
            raw_dt = raw.get("datetime", "").strip()
            try:
                dt_obj = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=UTC)
                dt_str = dt_obj.isoformat()
            except (ValueError, AttributeError):
                logger.warning("Row %d: unparseable datetime %r — skipped.", i, raw_dt)
                continue
            feed = raw.get("feed_source", "").strip() or "backtest"
            rows.append(_HeadlineRow(dt=dt_str, headline=headline, feed_source=feed))
    return rows


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay(
    rows: list[_HeadlineRow],
    *,
    dry_run: bool,
    max_cost_usd: float,
    skip_outcomes: bool,
) -> dict:
    """Run each headline through the pipeline and optionally fetch outcomes.

    Returns a summary dict with keys:
        total_headlines, passed_s1, passed_s2, signals_generated,
        total_cost_usd, cost_aborted (bool), outcomes_fetched.
    """
    total = len(rows)
    passed_s1 = 0
    passed_s2_or_vip = 0
    signals_generated = 0
    total_cost = 0.0
    cost_aborted = False
    outcomes_fetched = 0

    for idx, row in enumerate(rows, start=1):
        if total_cost >= max_cost_usd:
            logger.warning(
                "Cost limit $%.4f reached after %d/%d headlines — aborting replay.",
                max_cost_usd,
                idx - 1,
                total,
            )
            cost_aborted = True
            break

        hl_lower = row.headline.lower()
        tier = pipeline.stage1_matches(hl_lower)
        if tier is None:
            logger.debug("[%d/%d] S1 BLOCK: %s", idx, total, row.headline[:80])
            continue
        passed_s1 += 1

        logger.info(
            "[%d/%d] S1 PASS (%s): %s", idx, total, tier, row.headline[:80]
        )

        signals = pipeline.analyze(row.headline, feed_source=row.feed_source)

        if not signals:
            logger.info("  → No signals produced.")
            continue

        passed_s2_or_vip += 1

        for sig in signals:
            signals_generated += 1
            cost = sig.est_cost_usd or 0.0
            total_cost += cost
            if not dry_run:
                _db.insert_signal(sig)
            logger.info(
                "  → Signal: %s %s @ %d%% | cost=~$%.5f | %s",
                sig.ticker,
                sig.direction,
                sig.confidence,
                cost,
                sig.rationale[:80],
            )

        if dry_run or skip_outcomes:
            continue

        # Fetch outcomes for signals just generated.  We use a small sleep
        # buffer is unnecessary here — fetch_prices_for_signal handles the
        # "signal_too_recent" case gracefully and marks those as pending.
        for sig in signals:
            prices = fetch_prices_for_signal(
                ticker=sig.ticker,
                direction=sig.direction,
                created_utc=row.dt,
            )
            p0 = prices.get("price_at_signal")
            if p0 is not None:
                outcomes_fetched += 1
                logger.debug(
                    "  Outcome %s: entry=%.2f 1h_pnl=%s 24h_pnl=%s",
                    sig.ticker,
                    p0,
                    prices.get("outcome_pnl_1h"),
                    prices.get("outcome_pnl_24h"),
                )

    return {
        "total_headlines": total,
        "passed_s1": passed_s1,
        "passed_s2_or_vip": passed_s2_or_vip,
        "signals_generated": signals_generated,
        "total_cost_usd": total_cost,
        "cost_aborted": cost_aborted,
        "outcomes_fetched": outcomes_fetched,
    }


# ---------------------------------------------------------------------------
# Performance report
# ---------------------------------------------------------------------------


def performance_report(
    *,
    min_confidence: int = 0,
    ticker_filter: str | None = None,
    since_days: int = 90,
) -> None:
    """Print a structured performance report to stdout from DB signals.

    Only includes signals that have both ``price_at_signal`` and
    ``outcome_pnl_24h`` populated (i.e. outcomes have been fetched).
    """
    conn = _db.get_db()

    cutoff = (datetime.now(UTC) - timedelta(days=since_days)).isoformat()
    query = """
        SELECT ticker, direction, confidence, outcome_pnl_1h, outcome_pnl_24h,
               outcome_pnl_3d, outcome_pnl_7d, headline, created_utc
        FROM signals
        WHERE price_at_signal IS NOT NULL
          AND outcome_pnl_24h IS NOT NULL
          AND confidence >= ?
          AND created_utc >= ?
    """
    params: list = [min_confidence, cutoff]

    if ticker_filter:
        query += " AND ticker = ?"
        params.append(ticker_filter.upper())

    query += " ORDER BY created_utc DESC"

    rows = conn.execute(query, params).fetchall()
    rows = [dict(r) for r in rows]

    if not rows:
        print("No signals with completed outcome data match the criteria.")
        print(f"  min_confidence={min_confidence}, ticker={ticker_filter}, since_days={since_days}")
        print("\nRun 'python -m scripts.fetch_outcomes' to backfill missing outcome prices.")
        return

    # ------------------------------------------------------------------ #
    # Aggregate stats
    # ------------------------------------------------------------------ #
    total = len(rows)
    pnl_1h_vals = [r["outcome_pnl_1h"] for r in rows if r["outcome_pnl_1h"] is not None]
    pnl_24h_vals = [r["outcome_pnl_24h"] for r in rows]
    pnl_3d_vals = [r["outcome_pnl_3d"] for r in rows if r["outcome_pnl_3d"] is not None]
    pnl_7d_vals = [r["outcome_pnl_7d"] for r in rows if r["outcome_pnl_7d"] is not None]

    # Use the longest available horizon as primary metric.
    primary_label = "7d" if pnl_7d_vals else ("3d" if pnl_3d_vals else "24h")
    primary_vals = pnl_7d_vals or pnl_3d_vals or pnl_24h_vals
    primary_total = len(primary_vals)

    wins = sum(1 for p in primary_vals if p > 0)
    losses = sum(1 for p in primary_vals if p < 0)
    flat = primary_total - wins - losses

    avg_pnl_1h = sum(pnl_1h_vals) / len(pnl_1h_vals) if pnl_1h_vals else 0.0
    avg_pnl_24h = sum(pnl_24h_vals) / len(pnl_24h_vals) if pnl_24h_vals else 0.0
    avg_pnl_3d = sum(pnl_3d_vals) / len(pnl_3d_vals) if pnl_3d_vals else 0.0
    avg_pnl_7d = sum(pnl_7d_vals) / len(pnl_7d_vals) if pnl_7d_vals else 0.0

    # ------------------------------------------------------------------ #
    # Confidence calibration buckets — use primary horizon
    # ------------------------------------------------------------------ #
    # Build a mapping of confidence bucket → list of primary-horizon PnL.
    # For each signal we pick the best available horizon: 7d > 3d > 24h.
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        c = r["confidence"]
        if c >= 90:
            label = "90-100%"
        elif c >= 80:
            label = "80-89%"
        elif c >= 70:
            label = "70-79%"
        elif c >= 60:
            label = "60-69%"
        else:
            label = "<60%"
        pnl = r["outcome_pnl_7d"] or r["outcome_pnl_3d"] or r["outcome_pnl_24h"]
        if pnl is not None:
            buckets[label].append(pnl)

    # ------------------------------------------------------------------ #
    # Per-ticker breakdown — use primary horizon
    # ------------------------------------------------------------------ #
    by_ticker: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        pnl = r["outcome_pnl_7d"] or r["outcome_pnl_3d"] or r["outcome_pnl_24h"]
        if pnl is not None:
            by_ticker[r["ticker"]].append(pnl)

    # ------------------------------------------------------------------ #
    # Direction breakdown — use primary horizon
    # ------------------------------------------------------------------ #
    by_dir: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        pnl = r["outcome_pnl_7d"] or r["outcome_pnl_3d"] or r["outcome_pnl_24h"]
        if pnl is not None:
            by_dir[r["direction"]].append(pnl)

    # ------------------------------------------------------------------ #
    # Print
    # ------------------------------------------------------------------ #
    sep = "─" * 60
    print(f"\n{'═' * 60}")
    print(" AI-EDT SIGNAL PERFORMANCE REPORT")
    print(f" Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'═' * 60}")
    print(f" Criteria: conf≥{min_confidence}%"
          f"{', ticker=' + ticker_filter if ticker_filter else ''}"
          f", last {since_days} days")
    print(sep)

    print(f"\n{'OVERALL':}")
    print(f"  Signals with outcomes : {total}")
    print(f"  Win / Loss / Flat ({primary_label}): {wins} / {losses} / {flat}")
    win_rate = wins / primary_total * 100 if primary_total else 0
    print(f"  Win rate ({primary_label})          : {win_rate:.1f}%")
    print(f"  Avg PnL 1h             : {avg_pnl_1h:+.3%}")
    print(f"  Avg PnL 24h            : {avg_pnl_24h:+.3%}")
    if pnl_3d_vals:
        print(f"  Avg PnL 3d  (n={len(pnl_3d_vals):>3})   : {avg_pnl_3d:+.3%}")
    if pnl_7d_vals:
        print(f"  Avg PnL 7d  (n={len(pnl_7d_vals):>3})   : {avg_pnl_7d:+.3%}")

    print(f"\n{'CONFIDENCE CALIBRATION (' + primary_label + ' win rate)':}")
    bucket_order = ["90-100%", "80-89%", "70-79%", "60-69%", "<60%"]
    for label in bucket_order:
        vals = buckets.get(label)
        if not vals:
            continue
        wr = sum(1 for v in vals if v > 0) / len(vals) * 100
        avg = sum(vals) / len(vals)
        print(f"  {label:8s}  n={len(vals):3d}  win={wr:5.1f}%  avg={avg:+.3%}")

    print(f"\n{'DIRECTION BREAKDOWN (' + primary_label + ')':}")
    for direction in ("LONG", "SHORT"):
        vals = by_dir.get(direction, [])
        if not vals:
            continue
        wr = sum(1 for v in vals if v > 0) / len(vals) * 100
        avg = sum(vals) / len(vals)
        print(f"  {direction:5s}  n={len(vals):3d}  win={wr:5.1f}%  avg={avg:+.3%}")

    print(f"\n{'PER-TICKER BREAKDOWN (' + primary_label + ', sorted by n)':}")
    sorted_tickers = sorted(by_ticker.items(), key=lambda kv: -len(kv[1]))
    print(f"  {'Ticker':<8} {'n':>4} {'Win%':>7} {'Avg PnL':>10}")
    print(f"  {'-'*7} {'-'*4} {'-'*7} {'-'*10}")
    for tkr, vals in sorted_tickers:
        wr = sum(1 for v in vals if v > 0) / len(vals) * 100
        avg = sum(vals) / len(vals)
        print(f"  {tkr:<8} {len(vals):>4} {wr:>6.1f}% {avg:>+10.3%}")

    print(f"\n{'═' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.backtest",
        description="Phase 6 backtesting engine — headline replay and performance report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", type=Path, metavar="PATH", help="CSV file of headlines to replay.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline but do not persist signals or fetch outcomes.",
    )
    p.add_argument(
        "--max-cost",
        type=float,
        default=2.00,
        dest="max_cost",
        metavar="USD",
        help="Abort if cumulative estimated API cost exceeds this value.",
    )
    p.add_argument(
        "--skip-outcomes",
        action="store_true",
        dest="skip_outcomes",
        help="Do not attempt to fetch price outcomes after replay.",
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="Print performance report from existing DB signals (no replay).",
    )
    p.add_argument(
        "--min-confidence",
        type=int,
        default=0,
        dest="min_confidence",
        metavar="INT",
        help="Minimum confidence to include in the performance report.",
    )
    p.add_argument(
        "--ticker",
        type=str,
        default=None,
        metavar="TICKER",
        help="Filter report to a single ticker.",
    )
    p.add_argument(
        "--since",
        type=int,
        default=90,
        metavar="DAYS",
        help="Include only signals from the last N days in the report.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.csv and not args.report:
        parser.error("Provide --csv PATH to replay headlines, or --report for a DB summary.")

    if args.csv:
        if not args.csv.exists():
            print(f"Error: CSV file not found: {args.csv}", file=sys.stderr)
            return 1
        rows = load_csv(args.csv)
        if not rows:
            print("No valid rows found in CSV — nothing to replay.")
            return 0

        print(f"Loaded {len(rows)} headlines from {args.csv.name}")
        if args.dry_run:
            print("DRY RUN — signals will not be persisted and outcomes will not be fetched.")
        print(f"Cost limit: ${args.max_cost:.2f}\n")

        summary = replay(
            rows,
            dry_run=args.dry_run,
            max_cost_usd=args.max_cost,
            skip_outcomes=args.skip_outcomes,
        )

        print("\n── Replay summary ──────────────────────────────────────")
        print(f"  Headlines loaded      : {summary['total_headlines']}")
        print(f"  Passed Stage 1        : {summary['passed_s1']}")
        print(f"  Produced ≥1 signal    : {summary['passed_s2_or_vip']}")
        print(f"  Total signals         : {summary['signals_generated']}")
        print(f"  Estimated cost        : ${summary['total_cost_usd']:.5f}")
        if summary["cost_aborted"]:
            print("  ⚠ Cost limit reached — replay aborted early.")
        if not args.skip_outcomes and not args.dry_run:
            print(f"  Outcomes fetched      : {summary['outcomes_fetched']}")
        print("────────────────────────────────────────────────────────\n")

    if args.report:
        performance_report(
            min_confidence=args.min_confidence,
            ticker_filter=args.ticker,
            since_days=args.since,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
