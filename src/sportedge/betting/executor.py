"""Order execution. PaperExecutor logs intended fills; KalshiLiveExecutor sends
real signed orders to Kalshi and is only built when live mode is fully enabled.

Three independent switches are required before any real order can be sent:
  1. config.mode == "live"
  2. config.confirm_live is True   (both checked by config.live_enabled)
  3. complete Kalshi secrets present
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from sportedge.betting.strategy import Order
from sportedge.config import Config, Secrets


@dataclass
class Fill:
    ts: float
    side: str
    size: float
    price: float
    model_p: float
    edge: float
    mode: str
    token_id: str = ""


class PaperExecutor:
    """No network. Records what we *would* have done and tracks staked exposure."""

    mode = "paper"

    def __init__(self) -> None:
        self.fills: list[Fill] = []

    @property
    def staked(self) -> float:
        return sum(f.size for f in self.fills)

    def place(self, order: Order, token_id: str = "") -> Fill:
        fill = Fill(
            ts=time.time(),
            side=order.side,
            size=order.size,
            price=order.price,
            model_p=order.model_p,
            edge=order.edge,
            mode=self.mode,
            token_id=token_id,
        )
        self.fills.append(fill)
        return fill


class KalshiLiveExecutor(PaperExecutor):
    """Sends real signed orders to the Kalshi exchange. Inherits paper bookkeeping.

    Built only when every safety switch is on AND Kalshi keys are present.
    A stake in USDC becomes whole YES contracts inside the client.
    """

    mode = "live"

    def __init__(self, secrets: Secrets) -> None:
        super().__init__()
        if not secrets.kalshi_complete:
            raise ValueError("KalshiLiveExecutor requires complete Kalshi secrets")
        self.secrets = secrets
        self._client = None

    def _kalshi(self):
        if self._client is None:
            from sportedge.market.kalshi import KalshiClient

            self._client = KalshiClient(self.secrets.kalshi_host, self.secrets)
        return self._client

    def place(self, order: Order, token_id: str = "") -> Fill:
        self._kalshi().place_order(token_id, order.size, order.price)
        return super().place(order, token_id)


def make_executor(config: Config, secrets: Secrets) -> PaperExecutor:
    """Returns a Kalshi live executor only when every safety switch is on AND Kalshi
    keys are present; otherwise paper."""
    if config.live_enabled and secrets.kalshi_complete:
        return KalshiLiveExecutor(secrets)
    return PaperExecutor()
