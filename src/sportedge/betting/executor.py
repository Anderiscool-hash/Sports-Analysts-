"""Order execution. PaperExecutor logs intended fills; KalshiLiveExecutor sends
real signed orders to Kalshi and is only built when live mode is fully enabled.

Four independent switches are required before any real order can be sent:
  1. config.mode == "live"
  2. config.confirm_live is True   (both checked by config.live_enabled)
  3. complete Kalshi secrets present
  4. paper ledger proof passes config.paper_gate
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
    event_id: str = ""
    sport: str = ""
    league: str = ""
    home_team: str = ""
    away_team: str = ""
    selected_team: str = ""


class PaperExecutor:
    """No network. Records what we *would* have done and tracks staked exposure."""

    mode = "paper"

    def __init__(self, ledger_path: str | None = None) -> None:
        self.fills: list[Fill] = []
        self.ledger_path = ledger_path

    @property
    def staked(self) -> float:
        return sum(f.size for f in self.fills)

    def place(
        self,
        order: Order,
        token_id: str = "",
        metadata: dict[str, str] | None = None,
    ) -> Fill:
        metadata = metadata or {}
        fill = Fill(
            ts=time.time(),
            side=order.side,
            size=order.size,
            price=order.price,
            model_p=order.model_p,
            edge=order.edge,
            mode=self.mode,
            token_id=token_id,
            event_id=metadata.get("event_id", ""),
            sport=metadata.get("sport", ""),
            league=metadata.get("league", ""),
            home_team=metadata.get("home_team", ""),
            away_team=metadata.get("away_team", ""),
            selected_team=metadata.get("selected_team", ""),
        )
        self.fills.append(fill)
        if self.ledger_path:
            from sportedge.betting.paper import PaperLedger

            PaperLedger(self.ledger_path).append(fill)
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

    def place(
        self,
        order: Order,
        token_id: str = "",
        metadata: dict[str, str] | None = None,
    ) -> Fill:
        self._kalshi().place_order(token_id, order.size, order.price)
        return super().place(order, token_id, metadata=metadata)


def paper_gate_status(config: Config, paper_ledger_path: str | None = None) -> tuple[bool, str]:
    """Whether the persistent paper ledger has enough proof to allow live mode."""
    gate = config.paper_gate
    if not gate.enabled:
        return True, "paper proof gate disabled"
    if not paper_ledger_path:
        return False, "paper proof gate needs a ledger path"
    try:
        from sportedge.betting.paper import PaperLedger
        from sportedge.betting.report import collect_all_settlements

        ledger = PaperLedger(paper_ledger_path)
        fills = ledger.load()
        summary = ledger.summary(settlements=collect_all_settlements(fills))
    except Exception as exc:  # noqa: BLE001 - fail closed
        return False, f"paper proof gate could not read ledger: {exc}"
    fills = int(summary.get("fills", 0))
    settled_fills = int(summary.get("settled_fills", 0))
    min_settled_fills = gate.min_settled_fills if gate.min_settled_fills is not None else gate.min_fills
    total_pnl = float(summary.get("total_pnl", 0.0))
    realized_pnl = float(summary.get("realized_pnl", 0.0))
    realized_roi = float(summary.get("realized_roi", 0.0))
    if fills < gate.min_fills:
        return False, f"paper proof needs {gate.min_fills} fills; found {fills}"
    if settled_fills < min_settled_fills:
        return False, f"paper proof needs {min_settled_fills} settled fills; found {settled_fills}"
    if realized_pnl < gate.min_realized_pnl:
        return False, (
            f"paper proof needs realized PnL >= {gate.min_realized_pnl:.2f}; "
            f"found {realized_pnl:.2f}"
        )
    if realized_roi < gate.min_realized_roi:
        return False, (
            f"paper proof needs realized ROI >= {gate.min_realized_roi:.1%}; "
            f"found {realized_roi:.1%}"
        )
    if total_pnl < gate.min_total_pnl:
        return False, f"paper proof needs PnL >= {gate.min_total_pnl:.2f}; found {total_pnl:.2f}"
    return (
        True,
        f"paper proof passed: {fills} fills, {settled_fills} settled, "
        f"realized PnL {realized_pnl:+.2f}, realized ROI {realized_roi:+.1%}, "
        f"total PnL {total_pnl:+.2f}",
    )


def make_executor(
    config: Config,
    secrets: Secrets,
    paper_ledger_path: str | None = None,
) -> PaperExecutor:
    """Returns live executor only when safety switches, keys, and paper proof pass."""
    proof_ok, _reason = paper_gate_status(config, paper_ledger_path)
    if config.live_enabled and secrets.kalshi_complete and proof_ok:
        return KalshiLiveExecutor(secrets)
    return PaperExecutor(ledger_path=paper_ledger_path)
