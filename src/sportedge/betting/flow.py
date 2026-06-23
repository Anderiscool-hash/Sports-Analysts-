"""Trade-feed confirmation: whale + contrarian-momentum signals (pure).

Adapted from Krypt-Trader's whale / momentum scanner, but repurposed: SportEdge's
trade originates from the *model* edge ([[edge.py BottomDetector]]); this module
only answers "does recent order flow corroborate buying YES at this bottom?". It
never originates a trade. Fully unit-testable: list of trade dicts in, signal out.

A "buy confirmation" requires either:
  - a *whale* trade in the window (someone moved real size), or
  - *contrarian momentum*: YES price sold off >= ``price_move_threshold`` over the
    window (an overreaction worth fading -- the core "snipe the bottom" thesis).

Kalshi trade records vary by API version; fields are read defensively:
  count: ``count`` | ``count_fp``
  price: ``yes_price_dollars`` ("0.4200") | ``yes_price`` (cents) | ``no_price*``
  side:  ``taker_side`` ("yes"/"no")
  time:  ``created_time`` (ISO 8601) | ``ts`` / ``created_ts`` (epoch s or ms)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sportedge.config import FlowConfig


@dataclass(frozen=True)
class FlowSignal:
    confirms_buy: bool
    whale: bool
    momentum_down: bool
    cluster: bool
    trade_count: int
    notional: float
    reason: str


def _to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trade_count(trade: dict) -> float:
    for key in ("count", "count_fp"):
        value = _to_float(trade.get(key))
        if value is not None:
            return value
    return 0.0


def trade_yes_price(trade: dict) -> float | None:
    """YES price of a trade as a [0, 1] probability, regardless of record shape."""
    for key in ("yes_price_dollars", "no_price_dollars"):
        value = _to_float(trade.get(key))
        if value is not None:
            # *_dollars are already in [0, 1] dollars per contract.
            return value if key.startswith("yes") else 1.0 - value
    for key in ("yes_price", "no_price"):
        value = _to_float(trade.get(key))
        if value is not None:
            prob = value / 100.0 if value > 1.0 else value  # cents -> prob
            return prob if key.startswith("yes") else 1.0 - prob
    price = _to_float(trade.get("price"))
    if price is not None:
        return price / 100.0 if price > 1.0 else price
    return None


def trade_ts(trade: dict) -> float | None:
    """Epoch seconds for a trade, or ``None`` if no parseable timestamp."""
    for key in ("ts", "created_ts", "ts_ms"):
        value = _to_float(trade.get(key))
        if value is not None:
            return value / 1000.0 if value > 1e11 else value  # ms -> s
    raw = trade.get("created_time")
    if isinstance(raw, str) and raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def detect_flow(
    trades: list[dict],
    cfg: FlowConfig,
    now: float | None = None,
) -> FlowSignal:
    """Whale / contrarian-momentum confirmation from a market's recent trades."""
    if not trades:
        return FlowSignal(False, False, False, False, 0, 0.0, "no trades")

    now = now if now is not None else datetime.now(timezone.utc).timestamp()
    window: list[tuple[float, float, float]] = []  # (ts, yes_price, notional)
    for t in trades:
        price = trade_yes_price(t)
        if price is None:
            continue
        count = _trade_count(t)
        ts = trade_ts(t)
        if ts is not None and ts < now - cfg.lookback_sec:
            continue
        window.append((ts if ts is not None else now, price, price * count))

    if not window:
        return FlowSignal(False, False, False, False, 0, 0.0, "no trades in window")

    window.sort(key=lambda row: row[0])  # oldest -> newest
    total_notional = sum(row[2] for row in window)
    biggest = max(row[2] for row in window)
    trade_count = len(window)

    whale = biggest >= cfg.whale_min_notional
    cluster = trade_count >= cfg.cluster_min_trades and total_notional >= cfg.cluster_min_notional
    price_change = window[-1][1] - window[0][1]
    momentum_down = price_change <= -cfg.price_move_threshold

    confirms_buy = whale or momentum_down
    reasons = []
    if whale:
        reasons.append(f"whale ${biggest:.0f}")
    if momentum_down:
        reasons.append(f"sell-off {price_change:+.3f}")
    if cluster:
        reasons.append(f"cluster x{trade_count} ${total_notional:.0f}")
    if not confirms_buy:
        reasons.append(f"no confirm (move {price_change:+.3f}, max ${biggest:.0f})")

    return FlowSignal(
        confirms_buy=confirms_buy,
        whale=whale,
        momentum_down=momentum_down,
        cluster=cluster,
        trade_count=trade_count,
        notional=round(total_notional, 2),
        reason=", ".join(reasons),
    )
