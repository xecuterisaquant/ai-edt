"""Phase 3 Outcome Tracking — fetch post-signal prices and calculate PnL.

For each signal (ticker, direction, created_utc) we download 1h OHLC bars
from yfinance and populate four price columns:

    price_at_signal  — closest bar at/just-before signal time
    price_1h         — bar ~1h after signal (if the offset has elapsed)
    price_4h         — bar ~4h after signal (if the offset has elapsed)
    price_24h        — bar ~24h after signal (if the offset has elapsed)

PnL is expressed as a decimal fraction (e.g. 0.02 = +2 %):

    LONG   pnl = (exit - entry) / entry
    SHORT  pnl = (entry - exit) / entry

The dict returned by ``fetch_prices_for_signal`` maps directly to the
column names accepted by ``db.update_signal_prices``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from ai_edt.logger import get_logger

logger = get_logger("outcomes")

# Maximum allowed gap between a target timestamp and the nearest available
# bar.  If no bar falls within this window the price is treated as missing.
_MAX_BAR_GAP = timedelta(hours=2)


def _pnl(entry: float, exit_: float, direction: str) -> float:
    """Return decimal PnL rounded to 6 dp.  Handles zero-entry safely."""
    if entry == 0.0:
        return 0.0
    raw = (exit_ - entry) / entry
    return round(raw if direction == "LONG" else -raw, 6)


def _parse_utc(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp and ensure UTC awareness."""
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def fetch_prices_for_signal(
    ticker: str,
    direction: str,
    created_utc: str,
    *,
    now: datetime | None = None,
    _yf=None,  # injectable for unit testing; real callers leave as None
) -> dict:
    """Fetch post-signal prices for *ticker* and return a price/outcome dict.

    Keys:  price_at_signal, price_1h, price_4h, price_24h,
           outcome_pnl_1h, outcome_pnl_24h, outcome_note.

    All values may be ``None`` when data is unavailable (market closed,
    offset not yet elapsed, yfinance error, etc.).
    """
    if _yf is None:
        import yfinance as yf  # lazy import — optional dependency

        _yf = yf

    now = now or datetime.now(UTC)
    signal_dt = _parse_utc(created_utc)

    result: dict = {
        "price_at_signal": None,
        "price_1h": None,
        "price_4h": None,
        "price_24h": None,
        "outcome_pnl_1h": None,
        "outcome_pnl_24h": None,
        "outcome_note": None,
    }

    if (now - signal_dt) < timedelta(minutes=30):
        result["outcome_note"] = "signal_too_recent"
        return result

    # Download 1h bars covering the full 24h outcome window.
    start = signal_dt - timedelta(hours=2)
    end = min(signal_dt + timedelta(hours=27), now + timedelta(hours=1))

    try:
        hist = _yf.Ticker(ticker).history(start=start, end=end, interval="1h", auto_adjust=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance error for %s: %s", ticker, exc)
        result["outcome_note"] = f"yfinance_error: {exc}"
        return result

    if hist.empty:
        result["outcome_note"] = "no_price_data"
        return result

    # Normalise index to UTC for comparison.
    if hist.index.tz is None:
        hist.index = hist.index.tz_localize("UTC")
    else:
        hist.index = hist.index.tz_convert("UTC")

    def _nearest(target: datetime) -> float | None:
        """Return the Close price of the bar closest to *target*, or None."""
        if target > now:
            return None  # offset has not elapsed yet
        diffs_s = np.abs((hist.index - target).total_seconds())
        idx = int(np.argmin(diffs_s))
        if diffs_s[idx] > _MAX_BAR_GAP.total_seconds():
            return None
        return float(hist.iloc[idx]["Close"])

    p0 = _nearest(signal_dt)
    if p0 is None:
        result["outcome_note"] = "no_price_at_signal"
        return result

    result["price_at_signal"] = p0

    p1 = _nearest(signal_dt + timedelta(hours=1))
    p4 = _nearest(signal_dt + timedelta(hours=4))
    p24 = _nearest(signal_dt + timedelta(hours=24))

    result["price_1h"] = p1
    result["price_4h"] = p4
    result["price_24h"] = p24

    if p1 is not None:
        result["outcome_pnl_1h"] = _pnl(p0, p1, direction)
    if p24 is not None:
        result["outcome_pnl_24h"] = _pnl(p0, p24, direction)

    return result
