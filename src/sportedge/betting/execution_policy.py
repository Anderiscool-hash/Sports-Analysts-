"""Order-book-aware limit pricing for live BUY-YES orders.

Pure and fully unit-testable: no network, no Kalshi client. The live executor
fetches a raw order book and a fair price, then asks this module for the exact
limit price (in cents) to submit.

Why this exists: SportEdge previously submitted orders at the blind signal price.
In a score-lag snipe the price is stale for only a moment, so we want to either
cross the spread to fill *now* (``limit_cross``) or rest at the midpoint for a
better price (``limit_mid``). Adapted from Krypt-Trader's ``_compute_limit_price``.

Kalshi exposes two order-book shapes; both are handled here:
  - legacy, cents: ``{"orderbook": {"yes": [[cents, qty], ...], "no": [...]}}``
  - current, dollars: ``{"orderbook_fp": {"yes_dollars": [["0.3580", "158"], ...],
    "no_dollars": [...]}}``  (price as a dollar string, e.g. "0.3580" = 35.8c)
A ``yes`` level is a resting bid to buy YES; the best YES *bid* is the highest of
them. To buy YES you cross against a ``no`` bid: the YES *ask* is ``100 - best_no``.
"""

from __future__ import annotations


def _best_level_cents(levels: object, *, as_dollars: bool) -> int | None:
    """Highest price (in cents, 1..99) among ``[price, qty]`` book levels.

    ``as_dollars`` converts dollar-string prices ("0.3580" -> 36c); otherwise the
    price is already in cents. Sub-cent prices are rounded to the nearest cent.
    """
    if not isinstance(levels, list) or not levels:
        return None
    best: int | None = None
    for level in levels:
        try:
            raw = float(level[0])
        except (TypeError, ValueError, IndexError):
            continue
        cents = int(round(raw * 100)) if as_dollars else int(round(raw))
        if 1 <= cents <= 99 and (best is None or cents > best):
            best = cents
    return best


def best_yes_bid_ask_cents(orderbook: dict | None) -> tuple[int | None, int | None]:
    """Return ``(best_yes_bid, best_yes_ask)`` in cents from a Kalshi order book.

    Accepts the wrapped envelope (``orderbook`` / ``orderbook_fp``) or the bare
    inner book, in either the cents or the dollar-string format. Either side may be
    ``None`` when the book is empty on that side.
    """
    if not isinstance(orderbook, dict):
        return None, None
    book = orderbook.get("orderbook_fp") or orderbook.get("orderbook") or orderbook
    if not isinstance(book, dict):
        return None, None

    if "yes_dollars" in book or "no_dollars" in book:
        yes_levels, no_levels, as_dollars = book.get("yes_dollars"), book.get("no_dollars"), True
    else:
        yes_levels, no_levels, as_dollars = book.get("yes"), book.get("no"), False

    yes_bid = _best_level_cents(yes_levels, as_dollars=as_dollars)
    best_no = _best_level_cents(no_levels, as_dollars=as_dollars)
    yes_ask = (100 - best_no) if best_no is not None else None
    return yes_bid, yes_ask


def _clamp_cents(value: int) -> int:
    return max(1, min(99, value))


def compute_limit_price_cents(
    orderbook: dict | None,
    fair_price_prob: float,
    style: str = "limit_cross",
    fallback_offset_cents: int = 2,
) -> int:
    """Limit price (cents, 1..99) for a BUY-YES order.

    ``style``:
      - ``market`` / ``limit_cross``: pay the best YES ask (cross the spread).
      - ``limit_mid``: rest at the midpoint of best YES bid and best YES ask.

    Falls back to ``round(fair_price_prob*100) + fallback_offset_cents`` whenever
    the relevant side of the book is missing, so a BUY still clears.
    """
    fair_cents = _clamp_cents(int(round(fair_price_prob * 100)))
    fallback = _clamp_cents(fair_cents + max(0, fallback_offset_cents))
    yes_bid, yes_ask = best_yes_bid_ask_cents(orderbook)

    if style == "limit_mid":
        if yes_bid is not None and yes_ask is not None:
            return _clamp_cents(int(round((yes_bid + yes_ask) / 2)))
        return fallback

    # "market" and "limit_cross" both take the ask.
    if yes_ask is not None:
        return _clamp_cents(yes_ask)
    return fallback
