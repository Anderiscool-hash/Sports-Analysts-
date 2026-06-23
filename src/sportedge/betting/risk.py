"""Runtime risk gate: exposure caps and circuit breakers for live entries.

Borrowed from Krypt-Trader's pre-trade guards (``_is_blocked_by_daily_risk``,
exposure / open-position / per-event checks) and adapted to SportEdge's in-memory
``Fill`` records. These checks gate *new* live entries only; paper mode skips them.

Everything here is pure given a ``PortfolioState`` snapshot, so it is fully unit
testable without a Kalshi client.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sportedge.config import RiskConfig


@dataclass
class PortfolioState:
    """Snapshot of current exposure used to evaluate one prospective order."""

    bankroll: float
    current_staked: float = 0.0
    open_token_ids: set[str] = field(default_factory=set)
    event_position_counts: dict[str, int] = field(default_factory=dict)
    realized_pnl_today: float = 0.0


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str


def state_from_fills(
    fills,
    bankroll: float,
    realized_pnl_today: float = 0.0,
) -> PortfolioState:
    """Build a :class:`PortfolioState` from in-memory fills (duck-typed records).

    Each fill is treated as an open position for the duration of the loop; within a
    single live game there is no intra-loop settlement, which matches how the
    existing executor accounts for ``staked``.
    """
    staked = 0.0
    open_tokens: set[str] = set()
    event_counts: Counter[str] = Counter()
    for f in fills:
        staked += float(getattr(f, "size", 0.0) or 0.0)
        token = getattr(f, "token_id", "") or ""
        if token:
            open_tokens.add(token)
        event = getattr(f, "event_id", "") or ""
        if event:
            event_counts[event] += 1
    return PortfolioState(
        bankroll=bankroll,
        current_staked=staked,
        open_token_ids=open_tokens,
        event_position_counts=dict(event_counts),
        realized_pnl_today=realized_pnl_today,
    )


def _parse_hhmm(value: str) -> int:
    """Minutes since midnight for an ``HH:MM`` string (defaults to 0 on garbage)."""
    try:
        hh, mm = value.split(":")
        return (int(hh) % 24) * 60 + (int(mm) % 60)
    except (ValueError, AttributeError):
        return 0


def within_trading_hours(cfg: RiskConfig, now: datetime | None = None) -> bool:
    """Whether ``now`` (UTC) maps into the configured local trading window."""
    if not cfg.trading_hours_enabled:
        return True
    now = now or datetime.now(timezone.utc)
    local = now + timedelta(minutes=cfg.trading_timezone_offset_min)
    minute = local.hour * 60 + local.minute
    start = _parse_hhmm(cfg.trading_hours_start)
    end = _parse_hhmm(cfg.trading_hours_end)
    if start <= end:
        return start <= minute <= end
    # Window wraps past midnight (e.g. 22:00 -> 02:00).
    return minute >= start or minute <= end


class RiskManager:
    """Evaluates one prospective order against the configured caps and breakers."""

    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def check(
        self,
        *,
        order_size: float,
        token_id: str,
        event_id: str,
        state: PortfolioState,
        now: datetime | None = None,
    ) -> RiskDecision:
        cfg = self.cfg
        if not cfg.enabled:
            return RiskDecision(True, "risk gate disabled")

        if not within_trading_hours(cfg, now):
            return RiskDecision(False, "outside trading hours")

        if state.realized_pnl_today <= cfg.daily_loss_cap:
            return RiskDecision(
                False,
                f"daily loss cap hit ({state.realized_pnl_today:+.2f} <= {cfg.daily_loss_cap:+.2f})",
            )
        if cfg.daily_take_profit > 0 and state.realized_pnl_today >= cfg.daily_take_profit:
            return RiskDecision(
                False,
                f"daily take-profit hit ({state.realized_pnl_today:+.2f} >= {cfg.daily_take_profit:+.2f})",
            )

        projected = state.current_staked + order_size
        exposure_cap = state.bankroll * cfg.max_total_exposure_fraction
        if projected > exposure_cap + 1e-9:
            return RiskDecision(
                False, f"exposure cap ({projected:.2f} > {exposure_cap:.2f})"
            )
        reserve_floor = state.bankroll * (1.0 - cfg.min_cash_reserve_fraction)
        if projected > reserve_floor + 1e-9:
            return RiskDecision(
                False, f"cash reserve floor ({projected:.2f} > {reserve_floor:.2f})"
            )

        is_new_token = token_id not in state.open_token_ids
        if is_new_token and len(state.open_token_ids) >= cfg.max_open_positions:
            return RiskDecision(
                False, f"max open positions ({cfg.max_open_positions})"
            )

        if event_id:
            event_count = state.event_position_counts.get(event_id, 0)
            if is_new_token and event_count >= cfg.max_positions_per_event:
                return RiskDecision(
                    False,
                    f"max positions per event ({cfg.max_positions_per_event}) for {event_id}",
                )

        return RiskDecision(True, "ok")
