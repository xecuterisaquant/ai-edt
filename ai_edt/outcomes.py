"""Phase 3 Outcome Tracking — fetch post-signal prices and calculate PnL.

For each signal (ticker, direction, created_utc) we download price bars
from yfinance and populate six price columns:

    price_at_signal  — closest bar at/just-before signal time
    price_1h         — bar ~1h after signal (if the offset has elapsed)
    price_4h         — bar ~4h after signal (if the offset has elapsed)
    price_24h        — bar ~24h after signal (if the offset has elapsed)
    price_3d         — bar ~72h (3 days) after signal
    price_7d         — bar ~168h (7 days) after signal

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

    Keys:  price_at_signal, price_1h, price_4h, price_24h, price_3d, price_7d,
           outcome_pnl_1h, outcome_pnl_24h, outcome_pnl_3d, outcome_pnl_7d,
           outcome_note.

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
        "price_3d": None,
        "price_7d": None,
        "outcome_pnl_1h": None,
        "outcome_pnl_24h": None,
        "outcome_pnl_3d": None,
        "outcome_pnl_7d": None,
        "outcome_note": None,
    }

    if (now - signal_dt) < timedelta(minutes=30):
        result["outcome_note"] = "signal_too_recent"
        return result

    # ------------------------------------------------------------------
    # Download 1h bars for short-term windows (0h → 24h)
    # ------------------------------------------------------------------
    start = signal_dt - timedelta(hours=2)
    end_1h = min(signal_dt + timedelta(hours=27), now + timedelta(hours=1))

    try:
        hist_1h = _yf.Ticker(ticker).history(
            start=start, end=end_1h, interval="1h", auto_adjust=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance error for %s (1h): %s", ticker, exc)
        result["outcome_note"] = f"yfinance_error: {exc}"
        return result

    if hist_1h.empty:
        result["outcome_note"] = "no_price_data"
        return result

    # Normalise index to UTC for comparison.
    if hist_1h.index.tz is None:
        hist_1h.index = hist_1h.index.tz_localize("UTC")
    else:
        hist_1h.index = hist_1h.index.tz_convert("UTC")

    def _nearest_1h(target: datetime) -> float | None:
        """Return the Close price of the 1h bar closest to *target*."""
        if target > now:
            return None
        diffs_s = np.abs((hist_1h.index - target).total_seconds())
        idx = int(np.argmin(diffs_s))
        if diffs_s[idx] > _MAX_BAR_GAP.total_seconds():
            return None
        return float(hist_1h.iloc[idx]["Close"])

    p0 = _nearest_1h(signal_dt)
    if p0 is None:
        result["outcome_note"] = "no_price_at_signal"
        return result

    result["price_at_signal"] = p0

    p1 = _nearest_1h(signal_dt + timedelta(hours=1))
    p4 = _nearest_1h(signal_dt + timedelta(hours=4))
    p24 = _nearest_1h(signal_dt + timedelta(hours=24))

    result["price_1h"] = p1
    result["price_4h"] = p4
    result["price_24h"] = p24

    if p1 is not None:
        result["outcome_pnl_1h"] = _pnl(p0, p1, direction)
    if p24 is not None:
        result["outcome_pnl_24h"] = _pnl(p0, p24, direction)

    # ------------------------------------------------------------------
    # Download daily bars for multi-day windows (3d, 7d)
    # ------------------------------------------------------------------
    needs_3d = (now - signal_dt) >= timedelta(days=3)
    needs_7d = (now - signal_dt) >= timedelta(days=7)

    if needs_3d:
        try:
            start_d = signal_dt - timedelta(days=1)
            end_d = min(signal_dt + timedelta(days=9), now + timedelta(days=1))
            hist_d = _yf.Ticker(ticker).history(
                start=start_d, end=end_d, interval="1d", auto_adjust=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance error for %s (1d): %s", ticker, exc)
            # Short-term data is still valid — just skip multi-day.
            return result

        if not hist_d.empty:
            if hist_d.index.tz is None:
                hist_d.index = hist_d.index.tz_localize("UTC")
            else:
                hist_d.index = hist_d.index.tz_convert("UTC")

            def _nearest_d(target: datetime) -> float | None:
                """Return the Close of the daily bar closest to *target*."""
                if target > now:
                    return None
                diffs_s = np.abs((hist_d.index - target).total_seconds())
                idx = int(np.argmin(diffs_s))
                # Allow up to 2 days gap for weekends/holidays.
                if diffs_s[idx] > timedelta(days=2).total_seconds():
                    return None
                return float(hist_d.iloc[idx]["Close"])

            p3d = _nearest_d(signal_dt + timedelta(days=3))
            if p3d is not None:
                result["price_3d"] = p3d
                result["outcome_pnl_3d"] = _pnl(p0, p3d, direction)

            if needs_7d:
                p7d = _nearest_d(signal_dt + timedelta(days=7))
                if p7d is not None:
                    result["price_7d"] = p7d
                    result["outcome_pnl_7d"] = _pnl(p0, p7d, direction)

    return result
