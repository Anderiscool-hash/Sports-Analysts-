"""Order execution. PaperExecutor logs intended fills; KalshiLiveExecutor sends
real signed orders to Kalshi and is only built when live mode is fully enabled.

Four independent switches are required before any real order can be sent:
  1. config.mode == "live"
  2. config.confirm_live is True   (both checked by config.live_enabled)
  3. complete Kalshi secrets present
  4. paper ledger proof passes config.paper_gate

The live path borrows Krypt-Trader's order mechanics: order-book-aware limit
pricing, a short fill-poll window, and auto-cancel of any unfilled remainder so a
score-lag snipe never fills *after* the market has already corrected. The recorded
fill reflects what actually executed (count + average price), not the quote.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from sportedge.betting.execution_policy import compute_limit_price_cents
from sportedge.betting.reconcile import is_terminal, parse_kalshi_order
from sportedge.betting.strategy import Order
from sportedge.config import Config, ExecutionConfig, Secrets


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
    # Live-only execution detail (not persisted to the paper ledger).
    order_id: str = ""
    status: str = ""           # "filled" | "partial" | "unfilled" | "paper"
    requested_count: int = 0
    filled_count: int = 0
    avg_price: float = 0.0


def _fill_metadata(metadata: dict[str, str] | None) -> dict[str, str]:
    metadata = metadata or {}
    return {
        "event_id": metadata.get("event_id", ""),
        "sport": metadata.get("sport", ""),
        "league": metadata.get("league", ""),
        "home_team": metadata.get("home_team", ""),
        "away_team": metadata.get("away_team", ""),
        "selected_team": metadata.get("selected_team", ""),
    }


class PaperExecutor:
    """No network. Records what we *would* have done and tracks staked exposure."""

    mode = "paper"

    def __init__(self, ledger_path: str | None = None) -> None:
        self.fills: list[Fill] = []
        self.ledger_path = ledger_path

    @property
    def staked(self) -> float:
        return sum(f.size for f in self.fills)

    def _record(self, fill: Fill) -> Fill:
        """Append a fill to memory and (when configured) the persistent ledger."""
        self.fills.append(fill)
        if self.ledger_path:
            from sportedge.betting.paper import PaperLedger

            PaperLedger(self.ledger_path).append(fill)
        return fill

    def place(
        self,
        order: Order,
        token_id: str = "",
        metadata: dict[str, str] | None = None,
    ) -> Fill:
        fill = Fill(
            ts=time.time(),
            side=order.side,
            size=order.size,
            price=order.price,
            model_p=order.model_p,
            edge=order.edge,
            mode=self.mode,
            token_id=token_id,
            status="paper",
            **_fill_metadata(metadata),
        )
        return self._record(fill)


class KalshiLiveExecutor(PaperExecutor):
    """Sends real signed orders to the Kalshi exchange. Inherits paper bookkeeping.

    Built only when every safety switch is on AND Kalshi keys are present. A stake
    in USDC becomes whole YES contracts at the order-book-aware limit price; the
    order is polled for fills until ``order_expiration_sec`` then any remainder is
    canceled. ``clock``/``sleep`` are injectable so the poll loop is testable.
    """

    mode = "live"

    def __init__(
        self,
        secrets: Secrets,
        execution: ExecutionConfig | None = None,
        *,
        clock=time.time,
        sleep=time.sleep,
    ) -> None:
        super().__init__()
        if not secrets.kalshi_complete:
            raise ValueError("KalshiLiveExecutor requires complete Kalshi secrets")
        self.secrets = secrets
        self.execution = execution or ExecutionConfig()
        self._client = None
        self._clock = clock
        self._sleep = sleep

    def _kalshi(self):
        if self._client is None:
            from sportedge.market.kalshi import KalshiClient

            self._client = KalshiClient(self.secrets.kalshi_host, self.secrets)
        return self._client

    def _client_order_id(self, token_id: str) -> str:
        return f"se-{token_id or 'na'}-{int(self._clock() * 1000)}"

    def _await_fill(self, client, order_id: str, initial_order: dict, requested: int):
        """Poll a resting order until terminal or expired, then cancel any rest."""
        parsed = parse_kalshi_order(initial_order)
        cfg = self.execution

        if cfg.order_expiration_sec <= 0:
            # Expiry disabled: do a single best-effort reconcile, leave any rest.
            if order_id and not is_terminal(parsed):
                try:
                    parsed = parse_kalshi_order(client.get_order(order_id))
                except Exception:  # noqa: BLE001 - keep the initial read
                    pass
            return parsed

        deadline = self._clock() + cfg.order_expiration_sec
        while order_id and not is_terminal(parsed) and self._clock() < deadline:
            self._sleep(cfg.fill_poll_seconds)
            try:
                parsed = parse_kalshi_order(client.get_order(order_id))
            except Exception:  # noqa: BLE001 - transient; keep polling until deadline
                continue

        if order_id and parsed.remaining > 0 and not is_terminal(parsed):
            try:
                client.cancel_order(order_id)
                parsed = parse_kalshi_order(client.get_order(order_id))
            except Exception:  # noqa: BLE001 - cancel/reconcile is best-effort
                pass
        return parsed

    def place(
        self,
        order: Order,
        token_id: str = "",
        metadata: dict[str, str] | None = None,
    ) -> Fill:
        client = self._kalshi()
        cfg = self.execution

        book = client.get_orderbook(token_id) if token_id else {}
        limit_cents = compute_limit_price_cents(
            book, order.price, cfg.order_style, cfg.cross_spread_fallback_offset_cents
        )
        limit_prob = max(0.01, min(0.99, limit_cents / 100.0))
        requested = max(1, int(order.size / limit_prob))

        resp = client.place_limit_order(
            token_id, requested, limit_cents, self._client_order_id(token_id)
        )
        order_obj = resp.get("order", resp) or {}
        order_id = str(order_obj.get("order_id") or order_obj.get("id") or "")

        parsed = self._await_fill(client, order_id, order_obj, requested)
        avg_prob = (
            parsed.avg_cents / 100.0 if parsed.avg_cents is not None else limit_prob
        )
        avg_prob = max(0.01, min(0.99, avg_prob))
        actual_size = round(avg_prob * parsed.filled, 4)
        if parsed.filled <= 0:
            status = "unfilled"
        elif parsed.remaining > 0:
            status = "partial"
        else:
            status = "filled"

        fill = Fill(
            ts=self._clock(),
            side=order.side,
            size=actual_size,
            price=avg_prob,
            model_p=order.model_p,
            edge=order.edge,
            mode=self.mode,
            token_id=token_id,
            order_id=order_id,
            status=status,
            requested_count=requested,
            filled_count=parsed.filled,
            avg_price=avg_prob,
            **_fill_metadata(metadata),
        )
        return self._record(fill)


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
        return KalshiLiveExecutor(secrets, config.execution)
    return PaperExecutor(ledger_path=paper_ledger_path)
