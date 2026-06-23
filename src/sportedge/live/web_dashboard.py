"""Local browser dashboard for SportEdge.

The server binds to localhost and is read-only by default. Demo live execution is
available only after explicit mode selection and a separate strategy-arm action.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import threading
import time
import webbrowser
from dataclasses import asdict, dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, urlsplit

from dotenv import set_key
from rich.console import Console

from sportedge.betting.executor import KalshiLiveExecutor
from sportedge.betting.flow import FlowSignal, detect_flow, trade_ts, trade_yes_price
from sportedge.betting.history import ExecutionHistory
from sportedge.betting.paper import PaperLedger
from sportedge.betting.report import collect_all_settlements
from sportedge.config import Config, Secrets, load_config, load_secrets
from sportedge.data.espn_live import get_game_detail, list_live_games
from sportedge.live.dashboard import (
    PaperTradingEngine,
    TrendTracker,
    auto_configure_kalshi_market,
    build_market_info,
    build_readout,
    load_model,
    outcome_specs,
    settlement_marks,
)
from sportedge.market.kalshi import KalshiClient
from sportedge.types import GameCandidate, LiveDetail

WEB_DIR = Path(__file__).with_name("web")
DEFAULT_LEDGER = "data/cache/paper_ledger.parquet"
DEFAULT_HISTORY = "data/cache/execution_history.parquet"


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def candidate_payload(candidate: GameCandidate) -> dict[str, object]:
    return {
        **asdict(candidate),
        "display": f"{candidate.away_team} @ {candidate.home_team}",
        "is_live": candidate.status == "in",
    }


def detail_payload(detail: LiveDetail) -> dict[str, object]:
    data = asdict(detail)
    data["status_label"] = {"in": "LIVE", "pre": "UPCOMING", "post": "FINAL"}.get(
        detail.status, detail.status.upper()
    )
    data["clock_label"] = detail.clock or (
        f"{detail.minute:g}'" if detail.sport == "soccer" and detail.minute else "—"
    )
    return data


def tracking_status(
    ticker: str,
    price: float | None,
    edge: float | None,
    min_edge: float,
    signal: str,
) -> tuple[str, str]:
    """Operator-facing state for one contract on the trade radar."""
    if not ticker:
        return "UNCONFIGURED", "No direct Kalshi contract is configured"
    if price is None:
        return "SEEKING QUOTE", "Contract found; waiting for a usable ask"
    if "BUY" in signal:
        return "TRIGGERED", "Bottom and risk checks produced a paper entry"
    if edge is not None and edge >= min_edge:
        return "WATCHING BOTTOM", "Edge qualifies; waiting for dip and rebound"
    return "BELOW EDGE", f"Needs at least {min_edge:.1%} model edge"


def aggregate_holdings(report) -> list[dict[str, object]]:
    """Aggregate open paper fills into contract-level positions."""
    holdings: dict[str, dict[str, object]] = {}
    for row in report.to_dict(orient="records"):
        if bool(row.get("is_settled")):
            continue
        ticker = str(row.get("token_id") or "")
        if not ticker:
            continue
        price = _number(row.get("price"))
        cost = _number(row.get("size"))
        shares = _number(row.get("shares"), cost / price if price > 0 else 0.0)
        mark_value = row.get("mark")
        try:
            mark = float(mark_value)
            if math.isnan(mark):
                mark = None
        except (TypeError, ValueError):
            mark = None
        position = holdings.setdefault(
            ticker,
            {
                "ticker": ticker,
                "side": str(row.get("side") or "BUY"),
                "event_id": str(row.get("event_id") or ""),
                "sport": str(row.get("sport") or ""),
                "home_team": str(row.get("home_team") or ""),
                "away_team": str(row.get("away_team") or ""),
                "selected_team": str(row.get("selected_team") or ""),
                "shares": 0.0,
                "cost_basis": 0.0,
                "fill_count": 0,
                "mark": mark,
            },
        )
        position["shares"] = _number(position["shares"]) + shares
        position["cost_basis"] = _number(position["cost_basis"]) + cost
        position["fill_count"] = int(position["fill_count"]) + 1
        if mark is not None:
            position["mark"] = mark

    payload: list[dict[str, object]] = []
    for position in holdings.values():
        shares = _number(position["shares"])
        cost = _number(position["cost_basis"])
        mark = position["mark"]
        market_value = shares * _number(mark) if mark is not None else None
        matchup = " @ ".join(
            part for part in (str(position["away_team"]), str(position["home_team"])) if part
        )
        payload.append(
            {
                **position,
                "matchup": matchup,
                "average_entry": cost / shares if shares else 0.0,
                "market_value": market_value,
                "unrealized_pnl": market_value - cost if market_value is not None else None,
                "status": "MARKED" if mark is not None else "OPEN · UNMARKED",
            }
        )
    return sorted(payload, key=lambda row: _number(row["cost_basis"]), reverse=True)


def portfolio_stats(report, summary: dict[str, float | int], holdings) -> dict[str, object]:
    """User-facing performance statistics derived from the marked ledger."""
    records = report.to_dict(orient="records")
    settled = [row for row in records if bool(row.get("is_settled"))]
    winners = sum(_number(row.get("pnl")) > 0 for row in settled)
    losers = sum(_number(row.get("pnl")) < 0 for row in settled)
    edges = [_number(row.get("edge")) for row in records if row.get("edge") is not None]
    stakes = [_number(row.get("size")) for row in records]
    staked = _number(summary.get("staked"))
    total_pnl = _number(summary.get("total_pnl"))
    marked_positions = sum(position.get("mark") is not None for position in holdings)
    return {
        "fills": len(records),
        "contracts_held": len(holdings),
        "marked_positions": marked_positions,
        "settled_fills": len(settled),
        "winners": winners,
        "losers": losers,
        "win_rate": winners / len(settled) if settled else None,
        "average_edge": sum(edges) / len(edges) if edges else None,
        "average_stake": sum(stakes) / len(stakes) if stakes else 0.0,
        "total_staked": staked,
        "open_exposure": _number(summary.get("open_exposure")),
        "realized_pnl": _number(summary.get("realized_pnl")),
        "realized_roi": _number(summary.get("realized_roi")),
        "unrealized_pnl": _number(summary.get("unrealized_pnl")),
        "total_pnl": total_pnl,
        "total_return": total_pnl / staked if staked else 0.0,
    }


def trade_history_payload(paper_report, live_history) -> list[dict[str, object]]:
    """Normalize paper and demo-live executions into one newest-first feed."""
    rows: list[dict[str, object]] = []
    for record in paper_report.to_dict(orient="records"):
        rows.append(
            {
                "ts": _number(record.get("ts")),
                "mode": "paper",
                "side": str(record.get("side") or "BUY"),
                "ticker": str(record.get("token_id") or ""),
                "event_id": str(record.get("event_id") or ""),
                "sport": str(record.get("sport") or ""),
                "matchup": " @ ".join(
                    str(value)
                    for value in (record.get("away_team"), record.get("home_team"))
                    if value and str(value) != "nan"
                ),
                "selection": str(record.get("selected_team") or ""),
                "stake": _number(record.get("size")),
                "price": _number(record.get("price")),
                "shares": _number(record.get("shares")),
                "model_probability": _number(record.get("model_p")),
                "edge": _number(record.get("edge")),
                "status": "SETTLED" if bool(record.get("is_settled")) else "PAPER",
                "order_id": "",
                "filled_count": None,
                "requested_count": None,
                "pnl": _number(record.get("pnl")),
            }
        )
    for record in live_history.to_dict(orient="records"):
        price = _number(record.get("avg_price"), _number(record.get("price")))
        stake = _number(record.get("size"))
        rows.append(
            {
                "ts": _number(record.get("ts")),
                "mode": "demo-live",
                "side": str(record.get("side") or "BUY"),
                "ticker": str(record.get("token_id") or ""),
                "event_id": str(record.get("event_id") or ""),
                "sport": str(record.get("sport") or ""),
                "matchup": " @ ".join(
                    str(value)
                    for value in (record.get("away_team"), record.get("home_team"))
                    if value and str(value) != "nan"
                ),
                "selection": str(record.get("selected_team") or ""),
                "stake": stake,
                "price": price,
                "shares": _number(record.get("filled_count")),
                "model_probability": _number(record.get("model_p")),
                "edge": _number(record.get("edge")),
                "status": str(record.get("status") or "UNKNOWN").upper(),
                "order_id": str(record.get("order_id") or ""),
                "filled_count": int(_number(record.get("filled_count"))),
                "requested_count": int(_number(record.get("requested_count"))),
                "pnl": None,
            }
        )
    return sorted(rows, key=lambda row: _number(row["ts"]), reverse=True)


def whale_flow_payload(
    label: str,
    ticker: str,
    trades: list[dict],
    cfg,
    now: float | None = None,
) -> tuple[dict[str, object], FlowSignal]:
    """UI payload plus strategy confirmation for one contract's recent trades."""
    now = now if now is not None else time.time()
    signal = detect_flow(trades, cfg, now=now)
    events: list[dict[str, object]] = []
    for trade in trades:
        price = trade_yes_price(trade)
        ts = trade_ts(trade)
        if price is None or (ts is not None and ts < now - cfg.lookback_sec):
            continue
        count = _number(trade.get("count"), _number(trade.get("count_fp")))
        notional = price * count
        events.append(
            {
                "ts": ts or now,
                "price": price,
                "count": count,
                "notional": notional,
                "side": str(trade.get("taker_side") or trade.get("side") or "unknown").upper(),
                "is_whale": notional >= cfg.whale_min_notional,
            }
        )
    events.sort(key=lambda row: _number(row["ts"]), reverse=True)
    return (
        {
            "label": label,
            "ticker": ticker,
            "signal": asdict(signal),
            "trades": events[:20],
            "biggest_notional": max((_number(row["notional"]) for row in events), default=0.0),
        },
        signal,
    )


def assess_whale_follow(
    flow_payload: dict[str, object],
    model_probability: float,
    market_price: float | None,
    min_edge: float,
    whale_min_notional: float,
) -> dict[str, object]:
    """Judge whether the model agrees with the largest recent whale print.

    This is an explainable score, not a separately trained predictor. The sports
    model supplies fair value; whale direction, size, and clustering supply context.
    """
    trades = list(flow_payload.get("trades") or [])
    whale_trades = [trade for trade in trades if trade.get("is_whale")]
    if not whale_trades or market_price is None:
        return {
            "verdict": "NO WHALE" if not whale_trades else "WAIT FOR QUOTE",
            "score": 0,
            "reason": (
                "No qualifying whale print" if not whale_trades else "Market quote unavailable"
            ),
            "supports_yes": False,
        }
    biggest = max(whale_trades, key=lambda trade: _number(trade.get("notional")))
    side = str(biggest.get("side") or "UNKNOWN").upper()
    yes_edge = model_probability - market_price
    aligned_edge = yes_edge if side == "YES" else -yes_edge if side == "NO" else 0.0
    size_ratio = _number(biggest.get("notional")) / max(whale_min_notional, 1.0)
    score = 30.0 + min(size_ratio, 2.0) / 2.0 * 25.0
    score += max(-1.0, min(1.0, aligned_edge / 0.10)) * 35.0
    if bool((flow_payload.get("signal") or {}).get("cluster")):
        score += 10.0
    score = int(round(max(0.0, min(100.0, score))))

    if side not in {"YES", "NO"}:
        verdict = "WATCH"
        reason = "Whale direction is unavailable"
    elif aligned_edge >= min_edge:
        verdict = f"FOLLOW {side}"
        reason = f"Model agrees with {side} whale by {aligned_edge:+.1%}"
    elif aligned_edge <= -min_edge:
        verdict = f"FADE {side}"
        reason = f"Model disagrees with {side} whale by {abs(aligned_edge):.1%}"
    else:
        verdict = "WATCH"
        reason = f"Model/whale alignment {aligned_edge:+.1%} is below threshold"
    supports_yes = verdict == "FOLLOW YES" or verdict == "FADE NO"
    return {
        "verdict": verdict,
        "score": score,
        "reason": reason,
        "supports_yes": supports_yes,
        "whale_side": side,
        "whale_notional": _number(biggest.get("notional")),
        "model_edge": yes_edge,
    }


@dataclass
class _Session:
    candidate: GameCandidate
    cfg: Config
    model: object
    tracker: TrendTracker
    paper: PaperTradingEngine
    last_detail: LiveDetail | None = None
    last_signal_at: float = 0.0


class WebDashboardService:
    """Thread-safe state adapter between the browser and existing analyzer code."""

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        ledger_path: str = DEFAULT_LEDGER,
        history_path: str = DEFAULT_HISTORY,
        env_path: str = ".env",
        private_key_path: str = "config/kalshi_private_key.pem",
    ) -> None:
        self.config_path = config_path
        self.ledger_path = ledger_path
        self.history_path = history_path
        self.env_path = Path(env_path)
        self.private_key_path = Path(private_key_path)
        self.cfg = load_config(config_path)
        self.secrets = load_secrets()
        self.client = KalshiClient(self.secrets.kalshi_host, self.secrets)
        self._lock = threading.RLock()
        self._games: list[GameCandidate] = []
        self._games_at = 0.0
        self._session: _Session | None = None
        self.strategy_enabled = False
        self.trading_mode = "paper"
        self.flow_confirm_enabled = False
        self._mark_cache: dict[str, tuple[float, float | None]] = {}
        self._settlements: dict[str, float] = {}
        self._settlements_at = 0.0
        self._account_cache: tuple[float, dict[str, object]] = (0.0, {})
        self._paper_report = PaperLedger(self.ledger_path).report()
        self._flow_cache: dict[str, tuple[float, dict[str, object], FlowSignal]] = {}
        self._global_flow_cache: tuple[float, list[dict[str, object]]] = (0.0, [])

    @staticmethod
    def _validate_host(host: str) -> str:
        host = host.strip().rstrip("/")
        parts = urlsplit(host)
        allowed = parts.hostname and (
            parts.hostname.endswith(".kalshi.com") or parts.hostname.endswith(".kalshi.co")
        )
        if parts.scheme != "https" or not allowed or not parts.path.endswith("/trade-api/v2"):
            raise ValueError("Use an HTTPS Kalshi API host ending in /trade-api/v2")
        return host

    def account(self, force: bool = False) -> dict[str, object]:
        cached_at, cached = self._account_cache
        if not self.secrets.kalshi_complete:
            return {
                "configured": False,
                "connected": False,
                "balance": None,
                "portfolio_value": None,
            }
        if not force and cached and time.monotonic() - cached_at < 10:
            return cached
        try:
            raw = self.client.get_balance()
            balance = _number(raw.get("balance_dollars"), _number(raw.get("balance")) / 100)
            account = {
                "configured": True,
                "connected": True,
                "balance": balance,
                "portfolio_value": _number(raw.get("portfolio_value")) / 100,
                "updated_ts": raw.get("updated_ts"),
            }
        except Exception:  # noqa: BLE001 - header degrades without leaking auth details
            account = {
                "configured": True,
                "connected": False,
                "balance": None,
                "portfolio_value": None,
            }
        self._account_cache = (time.monotonic(), account)
        return account

    def settings(self) -> dict[str, object]:
        key_id = self.secrets.kalshi_api_key_id or ""
        hint = f"••••{key_id[-4:]}" if key_id else ""
        return {
            "kalshi_host": self.secrets.kalshi_host,
            "api_key_hint": hint,
            "api_key_configured": bool(key_id),
            "private_key_configured": bool(self.secrets.kalshi_private_key_pem),
            "account": self.account(),
        }

    def save_settings(self, payload: dict[str, object]) -> dict[str, object]:
        host = self._validate_host(str(payload.get("kalshi_host") or self.secrets.kalshi_host))
        key_id = str(payload.get("api_key_id") or "").strip()
        if not key_id:
            key_id = self.secrets.kalshi_api_key_id or ""
        private_key = str(payload.get("private_key_pem") or "").strip()
        if not private_key:
            private_key = self.secrets.kalshi_private_key_pem or ""
        if not key_id or not private_key:
            raise ValueError("API key ID and private key are both required")

        try:
            from cryptography.hazmat.primitives.serialization import load_pem_private_key

            load_pem_private_key(private_key.encode(), password=None)
        except Exception as exc:  # noqa: BLE001 - present a safe validation message
            raise ValueError("Private key is not a valid unencrypted PEM key") from exc

        candidate_secrets = Secrets(
            kalshi_api_key_id=key_id,
            kalshi_private_key_pem=private_key,
            kalshi_host=host,
        )
        candidate = KalshiClient(host, candidate_secrets)
        try:
            candidate.get_balance()
        except Exception as exc:  # noqa: BLE001 - do not persist credentials that fail auth
            raise ValueError("Kalshi rejected these credentials or the selected host") from exc

        self.private_key_path.parent.mkdir(parents=True, exist_ok=True)
        self.private_key_path.write_text(private_key + "\n", encoding="utf-8")
        set_key(str(self.env_path), "KALSHI_API_KEY_ID", key_id)
        set_key(str(self.env_path), "KALSHI_PRIVATE_KEY_PATH", str(self.private_key_path.resolve()))
        set_key(str(self.env_path), "KALSHI_PRIVATE_KEY_PEM", "")
        set_key(str(self.env_path), "KALSHI_HOST", host)
        os.environ["KALSHI_API_KEY_ID"] = key_id
        os.environ["KALSHI_PRIVATE_KEY_PATH"] = str(self.private_key_path.resolve())
        os.environ.pop("KALSHI_PRIVATE_KEY_PEM", None)
        os.environ["KALSHI_HOST"] = host
        self.secrets = candidate_secrets
        self.client = candidate
        self._account_cache = (0.0, {})
        return self.settings()

    def _portfolio(
        self,
        selected_marks: dict[str, float],
        selected_settlements: dict[str, float],
    ) -> tuple[dict[str, float | int], list[dict[str, object]], dict[str, object]]:
        ledger = PaperLedger(self.ledger_path)
        fills = ledger.load()
        now = time.monotonic()
        if now - self._settlements_at >= 60:
            try:
                self._settlements = collect_all_settlements(fills)
                self._settlements_at = now
            except Exception:  # noqa: BLE001 - retain the last known settlement state
                pass
        settlements = {**self._settlements, **selected_settlements}
        marks = dict(selected_marks)
        if not fills.empty and "token_id" in fills.columns:
            token_ids = sorted({str(value) for value in fills["token_id"].dropna() if str(value)})
            for ticker in token_ids:
                if ticker in settlements or ticker in marks or ticker.isdigit():
                    continue
                cached_at, cached_price = self._mark_cache.get(ticker, (0.0, None))
                if now - cached_at >= 10:
                    try:
                        cached_price = self.client.get_price(ticker, "BUY")
                    except Exception:  # noqa: BLE001 - an unquoted holding stays unmarked
                        cached_price = None
                    self._mark_cache[ticker] = (now, cached_price)
                if cached_price is not None:
                    marks[ticker] = cached_price
        report = ledger.report(marks=marks, settlements=settlements)
        self._paper_report = report
        summary = ledger.summary(marks=marks, settlements=settlements)
        holdings = aggregate_holdings(report)
        return summary, holdings, portfolio_stats(report, summary, holdings)

    def trade_history(self) -> list[dict[str, object]]:
        return trade_history_payload(
            self._paper_report,
            ExecutionHistory(self.history_path).load(),
        )

    def _make_signal_engine(self, cfg: Config) -> PaperTradingEngine:
        cfg = cfg.model_copy(deep=True)
        cfg.flow.mode = "confirm" if self.flow_confirm_enabled else "off"
        if self.trading_mode != "live":
            return PaperTradingEngine(cfg, self.ledger_path, market_client=self.client)
        live_cfg = cfg
        balance = _number(self.account(force=True).get("balance"), live_cfg.bankroll)
        live_cfg.bankroll = max(1.0, balance)
        executor = KalshiLiveExecutor(self.secrets, live_cfg.execution)
        return PaperTradingEngine(
            live_cfg,
            self.ledger_path,
            executor=executor,
            history_path=self.history_path,
            market_client=self.client,
        )

    def set_flow_confirmation(self, enabled: bool) -> dict[str, object]:
        with self._lock:
            self.flow_confirm_enabled = bool(enabled)
            if self._session is not None:
                self._session.paper.cfg.flow.mode = (
                    "confirm" if self.flow_confirm_enabled else "off"
                )
        return {
            "flow_confirm_enabled": self.flow_confirm_enabled,
            "trading_mode": self.trading_mode,
        }

    def whale_flow(
        self,
        specs: list[tuple[str, str]],
    ) -> tuple[list[dict[str, object]], dict[str, FlowSignal]]:
        payloads: list[dict[str, object]] = []
        signals: dict[str, FlowSignal] = {}
        now = time.monotonic()
        for label, ticker in specs:
            if not ticker:
                continue
            cached_at, payload, signal = self._flow_cache.get(ticker, (0.0, {}, None))
            if signal is None or now - cached_at >= max(2, self.cfg.loop.poll_seconds):
                trades = self.client.get_trades(ticker, limit=100)
                payload, signal = whale_flow_payload(label, ticker, trades, self.cfg.flow)
                self._flow_cache[ticker] = (now, payload, signal)
            payloads.append(payload)
            signals[ticker] = signal
        return payloads, signals

    def global_whale_flow(self) -> list[dict[str, object]]:
        """Poll Kalshi's global live tape and group recent activity by ticker."""
        cached_at, cached = self._global_flow_cache
        now = time.monotonic()
        if cached_at and now - cached_at < max(2, self.cfg.loop.poll_seconds):
            return cached
        trades = self.client.get_trades(limit=1000)
        grouped: dict[str, list[dict]] = {}
        for trade in trades:
            ticker = str(trade.get("ticker") or "")
            if ticker:
                grouped.setdefault(ticker, []).append(trade)
        payloads: list[dict[str, object]] = []
        for ticker, ticker_trades in grouped.items():
            payload, _signal = whale_flow_payload(
                ticker,
                ticker,
                ticker_trades,
                self.cfg.flow,
            )
            if payload["trades"]:
                payload["model_assessment"] = {
                    "verdict": "NO MODEL MATCH",
                    "score": 0,
                    "reason": "This global contract is not mapped to the selected sports model",
                    "supports_yes": False,
                }
                payloads.append(payload)
        payloads.sort(key=lambda row: _number(row["biggest_notional"]), reverse=True)
        self._global_flow_cache = (now, payloads[:25])
        return self._global_flow_cache[1]

    def set_mode(self, mode: str) -> dict[str, object]:
        if mode not in {"paper", "live"}:
            raise ValueError("Trading mode must be paper or live")
        if mode == "live":
            host = urlsplit(self.secrets.kalshi_host).hostname or ""
            if not host.endswith(".demo.kalshi.co"):
                raise ValueError("Web live trading is restricted to Kalshi's demo environment")
            if not self.account(force=True).get("connected"):
                raise ValueError("Connect a valid Kalshi demo account before enabling live mode")
        with self._lock:
            self.trading_mode = mode
            self.strategy_enabled = False
            if self._session is not None:
                self._session.paper = self._make_signal_engine(self._session.cfg)
        return {
            "trading_mode": self.trading_mode,
            "flow_confirm_enabled": self.flow_confirm_enabled,
            "strategy_enabled": self.strategy_enabled,
            "label": "Demo Live Trading" if mode == "live" else "Paper Trading",
        }

    def games(self, force: bool = False) -> list[GameCandidate]:
        with self._lock:
            if force or not self._games or time.monotonic() - self._games_at > 30:
                self._games = list_live_games()
                self._games_at = time.monotonic()
            return list(self._games)

    def bootstrap(self) -> dict[str, object]:
        games = self.games()
        ledger, _holdings, stats = self._portfolio({}, {})
        with self._lock:
            selected = self._session.candidate.event_id if self._session else None
        return {
            "app": "SportEdge",
            "mode": self.cfg.mode,
            "live_enabled": self.cfg.live_enabled,
            "paper_enabled": self.strategy_enabled,
            "strategy_enabled": self.strategy_enabled,
            "trading_mode": self.trading_mode,
            "poll_seconds": self.cfg.loop.poll_seconds,
            "selected_event_id": selected,
            "games": [candidate_payload(game) for game in games],
            "ledger": ledger,
            "stats": stats,
            "account": self.account(),
            "history": self.trade_history(),
        }

    def select(self, event_id: str) -> dict[str, object]:
        candidate = next((game for game in self.games() if game.event_id == event_id), None)
        if candidate is None:
            raise ValueError("Game is no longer available")
        cfg = self.cfg.model_copy(deep=True)
        model_path = cfg.model.path if candidate.sport == "basketball" else cfg.soccer.model_path
        model = load_model(candidate.sport, model_path)
        quiet = Console(file=io.StringIO(), force_terminal=False)
        auto_configure_kalshi_market(cfg, candidate, self.client, quiet)
        tracker = TrendTracker()
        for label, ticker in outcome_specs(
            cfg, candidate.sport, candidate.home_team, candidate.away_team
        ):
            if ticker:
                try:
                    tracker.seed(label, self.client.get_candlesticks(ticker.split("-")[0], ticker))
                except Exception:  # noqa: BLE001 - historical trend is optional
                    pass
        with self._lock:
            self._session = _Session(
                candidate,
                cfg,
                model,
                tracker,
                self._make_signal_engine(cfg),
            )
        return {"selected": candidate_payload(candidate)}

    def set_paper(self, enabled: bool) -> dict[str, bool]:
        with self._lock:
            self.strategy_enabled = bool(enabled)
        return {
            "paper_enabled": self.strategy_enabled,
            "strategy_enabled": self.strategy_enabled,
        }

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            session = self._session
        if session is None:
            raise ValueError("Select a game first")

        fresh = get_game_detail(
            session.candidate.sport,
            session.candidate.league,
            session.candidate.event_id,
        )
        with self._lock:
            if fresh is not None:
                session.last_detail = fresh
            detail = session.last_detail
            if detail is None:
                raise RuntimeError("Live game details are temporarily unavailable")

            rows = build_readout(session.cfg, detail, session.model, self.client, session.tracker)
            market_rows = build_market_info(session.cfg, detail, self.client)
            specs = outcome_specs(
                session.cfg, detail.sport, detail.home_team, detail.away_team
            )
            selected_flow, flow_signals = self.whale_flow(specs)
            readout_by_label = {readout[0]: readout for readout in rows}
            for flow_row in selected_flow:
                readout = readout_by_label[str(flow_row["label"])]
                assessment = assess_whale_follow(
                    flow_row,
                    _number(readout[1]),
                    None if readout[2] is None else _number(readout[2]),
                    session.cfg.edge.min_edge,
                    session.cfg.flow.whale_min_notional,
                )
                flow_row["model_assessment"] = assessment
                ticker = str(flow_row["ticker"])
                signal = flow_signals[ticker]
                if signal.whale:
                    confirms = signal.momentum_down or bool(assessment["supports_yes"])
                    reason = f"{signal.reason}; model {assessment['verdict'].lower()}"
                    flow_signals[ticker] = replace(
                        signal,
                        confirms_buy=confirms,
                        reason=reason,
                    )
                    flow_row["signal"] = asdict(flow_signals[ticker])
            selected_tickers = {str(row["ticker"]) for row in selected_flow}
            whale_flow = selected_flow + [
                row
                for row in self.global_whale_flow()
                if str(row["ticker"]) not in selected_tickers
            ]
            marks = {
                ticker: row[2]
                for (_label, ticker), row in zip(specs, rows, strict=False)
                if ticker and row[2] is not None
            }
            signals: list[tuple[str, str, bool, str, str]] = []
            now = time.monotonic()
            if self.strategy_enabled and now - session.last_signal_at >= self.cfg.loop.poll_seconds:
                signals = session.paper.update(
                    detail,
                    rows,
                    session.candidate.event_id,
                    flow_signals=flow_signals,
                )
                session.last_signal_at = now
            elif not self.strategy_enabled:
                signals = [
                    (label, ticker or "-", False, "OFF", "strategy is disarmed")
                    for label, ticker in specs
                ]
            summary, holdings, stats = self._portfolio(
                marks,
                settlement_marks(session.cfg, detail),
            )

        market_by_ticker = {
            ticker: snapshot for _label, ticker, snapshot in market_rows if snapshot is not None
        }
        signal_by_label = {row[0]: row for row in signals}
        outcomes = []
        for (label, ticker), (_row_label, model_p, price, edge, trend) in zip(
            specs, rows, strict=False
        ):
            market = market_by_ticker.get(ticker)
            signal = signal_by_label.get(label)
            signal_name = signal[3] if signal else "WAIT"
            radar_state, radar_reason = tracking_status(
                ticker,
                price,
                edge,
                session.cfg.edge.min_edge,
                signal_name,
            )
            outcomes.append(
                {
                    "label": label,
                    "ticker": ticker or "—",
                    "model_probability": _number(model_p),
                    "market_price": None if price is None else _number(price),
                    "edge": None if edge is None else _number(edge),
                    "trend": trend,
                    "bid": market.yes_bid if market else None,
                    "ask": market.yes_ask if market else None,
                    "volume": market.volume if market else None,
                    "liquidity": market.liquidity if market else None,
                    "signal": signal_name,
                    "signal_reason": signal[4] if signal else "waiting for next sample",
                    "is_bottom": bool(signal[2]) if signal else False,
                    "tracking_state": radar_state,
                    "tracking_reason": radar_reason,
                    "is_tracked": bool(ticker),
                }
            )
        return {
            "stale": fresh is None,
            "updated_at": time.time(),
            "paper_enabled": self.strategy_enabled,
            "strategy_enabled": self.strategy_enabled,
            "trading_mode": self.trading_mode,
            "flow_confirm_enabled": self.flow_confirm_enabled,
            "whale_flow": whale_flow,
            "min_edge": session.cfg.edge.min_edge,
            "game": detail_payload(detail),
            "outcomes": outcomes,
            "ledger": summary,
            "positions": holdings,
            "stats": stats,
            "pnl_updated_at": time.time(),
            "account": self.account(),
            "history": self.trade_history(),
        }


class WebDashboardHandler(BaseHTTPRequestHandler):
    service: WebDashboardService

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def _static(self, name: str, content_type: str) -> None:
        path = WEB_DIR / name
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        try:
            if path == "/api/bootstrap":
                self._json(self.service.bootstrap())
            elif path == "/api/settings":
                self._json(self.service.settings())
            elif path == "/api/snapshot":
                self._json(self.service.snapshot())
            elif path == "/api/health":
                self._json({"ok": True})
            elif path in {"/", "/index.html"}:
                self._static("index.html", "text/html; charset=utf-8")
            elif path == "/app.js":
                self._static("app.js", "text/javascript; charset=utf-8")
            elif path == "/styles.css":
                self._static("styles.css", "text/css; charset=utf-8")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, RuntimeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except Exception as exc:  # noqa: BLE001 - return a useful local UI error
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/select":
                self._json(self.service.select(str(body.get("event_id", ""))))
            elif path == "/api/paper":
                self._json(self.service.set_paper(bool(body.get("enabled"))))
            elif path == "/api/mode":
                self._json(self.service.set_mode(str(body.get("mode", ""))))
            elif path == "/api/flow":
                self._json(self.service.set_flow_confirmation(bool(body.get("enabled"))))
            elif path == "/api/settings":
                self._json(self.service.save_settings(body))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001 - return a useful local UI error
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def run(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    service = WebDashboardService()
    handler = type("SportEdgeHandler", (WebDashboardHandler,), {"service": service})
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    print(f"SportEdge UI running at {url} (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SportEdge local web dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    run(args.host, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
