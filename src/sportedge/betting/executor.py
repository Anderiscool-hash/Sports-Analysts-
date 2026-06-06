"""Order execution. PaperExecutor logs intended fills; LiveExecutor sends real
CLOB orders and is only built when live mode is fully enabled.

Three independent switches are required before any real order can be sent:
  1. config.mode == "live"
  2. config.confirm_live is True   (both checked by config.live_enabled)
  3. complete Polymarket secrets present
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


class LiveExecutor(PaperExecutor):
    """Sends real orders to the Polymarket CLOB. Inherits paper bookkeeping."""

    mode = "live"

    def __init__(self, secrets: Secrets) -> None:
        super().__init__()
        if not secrets.complete:
            raise ValueError("LiveExecutor requires complete Polymarket secrets")
        self.secrets = secrets
        self._client = None

    def _clob(self):
        if self._client is None:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            creds = None
            if self.secrets.api_key:
                creds = ApiCreds(
                    self.secrets.api_key,
                    self.secrets.api_secret,
                    self.secrets.api_passphrase,
                )
            self._client = ClobClient(
                self.secrets.clob_host,
                key=self.secrets.private_key,
                chain_id=self.secrets.chain_id,
                creds=creds,
                funder=self.secrets.funder_address,
            )
        return self._client

    def place(self, order: Order, token_id: str = "") -> Fill:
        from py_clob_client.clob_types import OrderArgs
        from py_clob_client.order_builder.constants import BUY

        # size here is USDC stake → shares = stake / price
        shares = round(order.size / order.price, 2)
        args = OrderArgs(price=order.price, size=shares, side=BUY, token_id=token_id)
        signed = self._clob().create_order(args)
        self._clob().post_order(signed)
        return super().place(order, token_id)


def make_executor(config: Config, secrets: Secrets) -> PaperExecutor:
    """Returns a LiveExecutor only when every safety switch is on; else paper."""
    if config.live_enabled and secrets.complete:
        return LiveExecutor(secrets)
    return PaperExecutor()
