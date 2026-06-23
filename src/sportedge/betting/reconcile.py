"""Fill reconciliation: turn a Kalshi order response into truth about what filled.

SportEdge's live executor used to *assume* a submitted order filled at the quoted
price. In a fast sports market that assumption silently corrupts PnL. This module
parses the authoritative order record so a recorded fill reflects what actually
executed. Adapted from Krypt-Trader's ``_parse_kalshi_order`` / poll loop.
"""

from __future__ import annotations

from dataclasses import dataclass


def _first_int(order: dict, *keys: str) -> int | None:
    for key in keys:
        value = order.get(key)
        if value in (None, ""):
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return None


@dataclass(frozen=True)
class ParsedOrder:
    filled: int            # contracts that actually executed
    remaining: int         # contracts still resting (0 if fully done)
    avg_cents: float | None  # average fill price in cents, when known
    cost_cents: int        # total spent in cents, when known
    status: str            # raw Kalshi status string, lowercased


def parse_kalshi_order(order: dict) -> ParsedOrder:
    """Normalize a Kalshi order object into fill counts and average price.

    Kalshi exposes counts under varying keys across endpoints; we try the common
    ones defensively so partial fills and full fills both reconcile correctly.
    """
    order = order or {}
    status = str(order.get("status") or "").lower()

    place = _first_int(order, "place_count", "initial_count", "count") or 0
    remaining = _first_int(order, "remaining_count", "remaining")
    filled = _first_int(order, "fill_count", "filled_count", "taker_fill_count")
    if filled is None and remaining is not None:
        filled = max(0, place - remaining)
    if filled is None:
        filled = 0
    if remaining is None:
        remaining = max(0, place - filled)

    avg_cents = None
    avg_raw = order.get("average_fill_price") or order.get("avg_fill_price")
    if avg_raw not in (None, ""):
        try:
            avg_cents = float(avg_raw)
        except (TypeError, ValueError):
            avg_cents = None
    if avg_cents is None and filled:
        # Fall back to the order's own limit price when no average is reported.
        limit = _first_int(order, "yes_price")
        if limit is not None:
            avg_cents = float(limit)

    cost = _first_int(order, "taker_fill_cost", "fill_cost")
    if cost is None and avg_cents is not None:
        cost = int(round(avg_cents * filled))
    cost = cost or 0

    return ParsedOrder(
        filled=filled,
        remaining=remaining,
        avg_cents=avg_cents,
        cost_cents=cost,
        status=status,
    )


def is_terminal(parsed: ParsedOrder) -> bool:
    """Whether an order needs no further polling (done, dead, or fully filled)."""
    if parsed.remaining <= 0 and parsed.filled > 0:
        return True
    return parsed.status in {"executed", "canceled", "cancelled", "expired"}
