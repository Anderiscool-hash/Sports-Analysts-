"""Edge math and the "snipe the bottom" detector.

Pure, dependency-free, fully unit-tested. A Kalshi YES price in [0, 1] is read
directly as the market-implied probability, so:

    edge = model_p - price

A positive edge means our model rates the outcome higher than the market is pricing
it — i.e. the YES share is cheap relative to fair value.
"""

from __future__ import annotations

from dataclasses import dataclass


def implied_prob_from_price(price: float) -> float:
    """Kalshi YES price already equals implied probability (0..1)."""
    return price


def edge(model_p: float, price: float) -> float:
    return model_p - price


@dataclass
class BottomSignal:
    is_bottom: bool
    price: float
    model_p: float
    edge: float
    reason: str = ""


class BottomDetector:
    """Tracks one token's price stream and flags a local bottom.

    A bottom fires when ALL hold:
      1. price fell from a recent peak by >= ``dip_threshold`` (a real dip),
      2. price has since ticked back up for ``rebound_ticks`` consecutive updates
         (the overshoot is reversing — we're past the low), and
      3. the model edge at the current price is still >= ``min_edge``.

    After firing it rebaselines so the same dip can't fire repeatedly.
    """

    def __init__(self, dip_threshold: float, min_edge: float, rebound_ticks: int = 1):
        self.dip_threshold = dip_threshold
        self.min_edge = min_edge
        self.rebound_ticks = max(1, rebound_ticks)
        self._peak: float | None = None
        self._trough: float | None = None
        self._prev: float | None = None
        self._rebound = 0

    def _rebaseline(self, price: float) -> None:
        self._peak = price
        self._trough = price
        self._prev = price
        self._rebound = 0

    def update(self, price: float, model_p: float) -> BottomSignal:
        e = edge(model_p, price)
        sig = BottomSignal(False, price, model_p, e)

        if self._peak is None:
            self._rebaseline(price)
            return sig

        # New local high → start a fresh episode from this peak.
        if price >= self._peak:
            self._rebaseline(price)
            sig.reason = "new peak"
            return sig

        # Below the peak: update trough / count rebound ticks.
        if price < self._trough:  # type: ignore[operator]
            self._trough = price
            self._rebound = 0
        elif price > self._prev:  # type: ignore[operator]
            self._rebound += 1
        # flat or small pullback that isn't a new low: leave rebound count as-is
        self._prev = price

        dip = self._peak - self._trough  # type: ignore[operator]
        if dip >= self.dip_threshold and self._rebound >= self.rebound_ticks and e >= self.min_edge:
            sig.is_bottom = True
            sig.reason = (
                f"dip {dip:.3f} from peak {self._peak:.3f}, "
                f"rebound x{self._rebound}, edge {e:.3f}"
            )
            self._rebaseline(price)
        return sig
