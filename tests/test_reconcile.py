"""Kalshi order fill parsing (pure)."""

from sportedge.betting.reconcile import is_terminal, parse_kalshi_order


def test_full_fill_from_explicit_counts():
    p = parse_kalshi_order(
        {"status": "executed", "place_count": 10, "fill_count": 10, "average_fill_price": 42}
    )
    assert p.filled == 10
    assert p.remaining == 0
    assert p.avg_cents == 42.0
    assert is_terminal(p)


def test_partial_fill_infers_filled_from_remaining():
    p = parse_kalshi_order(
        {"status": "resting", "place_count": 10, "remaining_count": 4, "yes_price": 41}
    )
    assert p.filled == 6
    assert p.remaining == 4
    assert p.avg_cents == 41.0  # falls back to limit price when no average reported
    assert not is_terminal(p)


def test_unfilled_resting_order():
    p = parse_kalshi_order({"status": "resting", "place_count": 5, "remaining_count": 5})
    assert p.filled == 0
    assert p.remaining == 5
    assert not is_terminal(p)


def test_canceled_is_terminal():
    p = parse_kalshi_order({"status": "canceled", "place_count": 5, "remaining_count": 5})
    assert p.filled == 0
    assert is_terminal(p)


def test_cost_derived_from_avg_and_filled_when_absent():
    p = parse_kalshi_order(
        {"status": "executed", "place_count": 3, "fill_count": 3, "average_fill_price": 50}
    )
    assert p.cost_cents == 150


def test_empty_order_is_safe():
    p = parse_kalshi_order({})
    assert p.filled == 0 and p.remaining == 0 and p.avg_cents is None
