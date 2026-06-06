"""Position sizing. Fractional-Kelly on a binary (YES at `price` pays 1 if it hits).

Refuses to size unless the signal is a confirmed bottom with edge >= min_edge.
"""

from __future__ import annotations

from dataclasses import dataclass

from sportedge.market.edge import BottomSignal


@dataclass
class Order:
    side: str  # "BUY"
    size: float  # stake in USDC
    price: float
    model_p: float
    edge: float


def kelly_stake(
    model_p: float,
    price: float,
    bankroll: float,
    kelly_fraction: float,
    max_stake: float,
) -> float:
    """Fractional-Kelly stake for buying a YES share at `price`.

    Net odds b = (1 - price) / price. Full-Kelly f* = p - (1 - p) / b.
    Returns a non-negative stake, capped by max_stake.
    """
    if not (0.0 < price < 1.0):
        return 0.0
    b = (1.0 - price) / price
    f_star = model_p - (1.0 - model_p) / b
    f = max(0.0, f_star) * kelly_fraction
    return min(f * bankroll, max_stake)


class Strategy:
    def __init__(self, min_edge: float, kelly_fraction: float, max_stake: float, bankroll: float):
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.max_stake = max_stake
        self.bankroll = bankroll

    def decide(self, signal: BottomSignal) -> Order | None:
        if not signal.is_bottom or signal.edge < self.min_edge:
            return None
        stake = kelly_stake(
            signal.model_p, signal.price, self.bankroll, self.kelly_fraction, self.max_stake
        )
        if stake <= 0.0:
            return None
        return Order("BUY", round(stake, 4), signal.price, signal.model_p, signal.edge)
