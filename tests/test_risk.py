"""Runtime risk gate: exposure caps + circuit breakers (pure)."""

from dataclasses import dataclass
from datetime import datetime, timezone

from sportedge.betting.risk import (
    PortfolioState,
    RiskManager,
    state_from_fills,
    within_trading_hours,
)
from sportedge.config import RiskConfig


@dataclass
class _FakeFill:
    size: float
    token_id: str
    event_id: str


def _mgr(**overrides) -> RiskManager:
    return RiskManager(RiskConfig(**overrides))


def test_allows_order_within_all_caps():
    mgr = _mgr()
    state = PortfolioState(bankroll=100.0, current_staked=0.0)
    assert mgr.check(order_size=5.0, token_id="T", event_id="E", state=state).allowed


def test_blocks_when_exposure_cap_exceeded():
    mgr = _mgr(max_total_exposure_fraction=0.50)
    state = PortfolioState(bankroll=100.0, current_staked=48.0)
    d = mgr.check(order_size=5.0, token_id="T", event_id="E", state=state)
    assert not d.allowed and "exposure cap" in d.reason


def test_blocks_when_cash_reserve_floor_breached():
    # exposure cap high, but reserve floor (90% of bankroll) is the binding limit
    mgr = _mgr(max_total_exposure_fraction=1.0, min_cash_reserve_fraction=0.10)
    state = PortfolioState(bankroll=100.0, current_staked=88.0)
    d = mgr.check(order_size=5.0, token_id="T", event_id="E", state=state)
    assert not d.allowed and "cash reserve" in d.reason


def test_blocks_new_token_when_max_open_positions_hit():
    mgr = _mgr(max_open_positions=2, max_positions_per_event=99)
    state = PortfolioState(bankroll=1000.0, open_token_ids={"A", "B"})
    d = mgr.check(order_size=1.0, token_id="C", event_id="E", state=state)
    assert not d.allowed and "max open positions" in d.reason
    # adding to an already-open token is still allowed
    assert mgr.check(order_size=1.0, token_id="A", event_id="E", state=state).allowed


def test_blocks_second_position_on_same_event():
    mgr = _mgr(max_positions_per_event=1)
    state = PortfolioState(
        bankroll=1000.0, open_token_ids={"A"}, event_position_counts={"E": 1}
    )
    d = mgr.check(order_size=1.0, token_id="B", event_id="E", state=state)
    assert not d.allowed and "per event" in d.reason


def test_daily_loss_cap_halts_new_entries():
    mgr = _mgr(daily_loss_cap=-20.0)
    state = PortfolioState(bankroll=1000.0, realized_pnl_today=-25.0)
    d = mgr.check(order_size=1.0, token_id="A", event_id="E", state=state)
    assert not d.allowed and "loss cap" in d.reason


def test_daily_take_profit_halts_new_entries():
    mgr = _mgr(daily_take_profit=50.0)
    state = PortfolioState(bankroll=1000.0, realized_pnl_today=60.0)
    d = mgr.check(order_size=1.0, token_id="A", event_id="E", state=state)
    assert not d.allowed and "take-profit" in d.reason


def test_disabled_gate_allows_everything():
    mgr = _mgr(enabled=False, max_total_exposure_fraction=0.0)
    state = PortfolioState(bankroll=100.0, current_staked=100.0)
    assert mgr.check(order_size=50.0, token_id="A", event_id="E", state=state).allowed


def test_trading_hours_window():
    cfg = RiskConfig(
        trading_hours_enabled=True,
        trading_hours_start="09:00",
        trading_hours_end="17:00",
        trading_timezone_offset_min=0,
    )
    inside = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
    outside = datetime(2026, 6, 21, 23, 0, tzinfo=timezone.utc)
    assert within_trading_hours(cfg, inside)
    assert not within_trading_hours(cfg, outside)


def test_state_from_fills_aggregates_exposure_and_counts():
    fills = [
        _FakeFill(5.0, "A", "E1"),
        _FakeFill(3.0, "B", "E1"),
        _FakeFill(2.0, "A", "E1"),
    ]
    state = state_from_fills(fills, bankroll=100.0)
    assert state.current_staked == 10.0
    assert state.open_token_ids == {"A", "B"}
    assert state.event_position_counts == {"E1": 3}
