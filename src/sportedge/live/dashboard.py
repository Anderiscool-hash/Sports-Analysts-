"""Live game dashboard — a rich, in-terminal watch screen.

Pick a live (or upcoming) game, then watch a constantly-updating panel: score,
clock, sport-specific detail (basketball fouls / free throws; soccer cards /
set-pieces), possession, the raw last play, and your model's win-probability with
a best-effort Kalshi price + edge alongside.

Safe by default: paper signals can be written to a persistent paper ledger; it
never sends live orders.

    python -m sportedge.live.dashboard
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sportedge.betting.executor import PaperExecutor
from sportedge.betting.paper import PaperLedger
from sportedge.betting.strategy import Strategy
from sportedge.config import Config, load_config, load_secrets
from sportedge.data.espn_live import get_game_detail, list_live_games
from sportedge.market.edge import BottomDetector
from sportedge.market.kalshi import KalshiClient, KalshiMarketSnapshot
from sportedge.market.scanner import GameMarketCoverage, scan_game_markets
from sportedge.model.live_winprob import WinProbModel
from sportedge.model.soccer_winprob import SoccerWinProbModel
from sportedge.types import (
    GameCandidate,
    GameState,
    LiveDetail,
    SoccerGameState,
    regulation_seconds_remaining,
)

# One readout row: outcome label, model P, market price, edge, price-trend sparkline.
ReadoutRow = tuple[str, float, float | None, float | None, str]
MarketInfoRow = tuple[str, str, KalshiMarketSnapshot | None]
PaperSignalRow = tuple[str, str, bool, str, str]

_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    """A unicode block sparkline for a price series. Flat/empty series degrade
    gracefully (single level / empty string)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return _SPARK_BLOCKS[0] * len(vals)
    span = hi - lo
    out = []
    for v in vals:
        idx = int((v - lo) / span * (len(_SPARK_BLOCKS) - 1))
        out.append(_SPARK_BLOCKS[idx])
    return "".join(out)


class TrendTracker:
    """Per-outcome price history for the trend sparkline. Seeded from Kalshi
    candlesticks (historical) and extended by the prices sampled each poll tick."""

    def __init__(self, maxlen: int = 60):
        from collections import deque

        self._deque = deque
        self._maxlen = maxlen
        self._history: dict[str, object] = {}

    def seed(self, label: str, prices: list[float]) -> None:
        if prices:
            self._history[label] = self._deque(prices[-self._maxlen :], maxlen=self._maxlen)

    def append(self, label: str, price: float | None) -> None:
        if price is None:
            return
        bucket = self._history.get(label)
        if bucket is None:
            bucket = self._history[label] = self._deque(maxlen=self._maxlen)
        bucket.append(price)

    def series(self, label: str) -> list[float]:
        return list(self._history.get(label, []))


def outcome_specs(cfg: Config, sport: str, home_team: str, away_team: str) -> list[tuple[str, str]]:
    """(outcome label, Kalshi ticker) pairs in display order for the sport."""
    if sport == "basketball":
        specs = [(f"{home_team} win", cfg.market.kalshi_ticker)]
        if cfg.market.kalshi_away_ticker:
            specs.append((f"{away_team} win", cfg.market.kalshi_away_ticker))
        return specs
    s = cfg.soccer
    return [
        (home_team, s.kalshi_home_ticker),
        ("Draw", s.kalshi_draw_ticker),
        (away_team, s.kalshi_away_ticker),
    ]


# ----- pure conversions (testable) -----
def clock_to_seconds(clock: str) -> float:
    """A "M:SS" / "MM:SS" display clock -> seconds left in the period. Soccer-style
    clocks like "67'" have no colon and read as 0 (basketball-only helper)."""
    match = re.match(r"(\d+):(\d+)", clock or "")
    if not match:
        return 0.0
    return int(match.group(1)) * 60 + int(match.group(2))


def detail_to_basketball_state(detail: LiveDetail, pre_game_home_prob: float = 0.60) -> GameState:
    period = detail.period or 1
    secs = regulation_seconds_remaining(period, clock_to_seconds(detail.clock))
    return GameState(
        home_team=detail.home_team,
        away_team=detail.away_team,
        home_score=detail.home_score,
        away_score=detail.away_score,
        period=period,
        seconds_remaining=secs,
        pre_game_home_prob=pre_game_home_prob,
    )


def detail_to_soccer_state(
    detail: LiveDetail, lambda_home: float, lambda_away: float
) -> SoccerGameState:
    return SoccerGameState(
        home_team=detail.home_team,
        away_team=detail.away_team,
        home_goals=detail.home_score,
        away_goals=detail.away_score,
        minute=detail.minute,
        home_red_cards=detail.home_red,
        away_red_cards=detail.away_red,
        lambda_home=lambda_home,
        lambda_away=lambda_away,
    )


def _safe_price(client: KalshiClient | None, ticker: str) -> float | None:
    """Best-effort Kalshi price read; None on no ticker or any error."""
    if client is None or not ticker:
        return None
    try:
        return client.get_price(ticker, "BUY")
    except Exception:  # noqa: BLE001 - market data is optional; degrade quietly
        return None


def build_market_info(
    cfg: Config,
    detail: LiveDetail,
    client: KalshiClient | None,
) -> list[MarketInfoRow]:
    """Best-effort Kalshi quote metadata per configured outcome."""
    if client is None:
        return []
    rows: list[MarketInfoRow] = []
    for label, ticker in outcome_specs(cfg, detail.sport, detail.home_team, detail.away_team):
        if not ticker:
            continue
        try:
            snapshot = client.get_market_snapshot(ticker)
        except Exception:  # noqa: BLE001 - market metadata is optional
            snapshot = None
        rows.append((label, ticker, snapshot))
    return rows


def build_readout(
    cfg: Config,
    detail: LiveDetail,
    model,
    client: KalshiClient | None,
    tracker: TrendTracker | None = None,
) -> list[ReadoutRow]:
    """Model win-prob + (best-effort) Kalshi price + edge + price trend per outcome.

    When a ``tracker`` is supplied, each tick's sampled price is appended to it and
    the row carries a sparkline of that outcome's price history."""
    specs = outcome_specs(cfg, detail.sport, detail.home_team, detail.away_team)
    if detail.sport == "basketball":
        home_p = model.predict(detail_to_basketball_state(detail))
        probs = [home_p]
        if len(specs) > 1:
            probs.append(1.0 - home_p)
    else:
        p = model.predict(
            detail_to_soccer_state(detail, cfg.soccer.lambda_home, cfg.soccer.lambda_away)
        )
        probs = [p.home, p.draw, p.away]

    rows: list[ReadoutRow] = []
    for (label, ticker), prob in zip(specs, probs):
        price = _safe_price(client, ticker)
        edge = (prob - price) if price is not None else None
        trend = ""
        if tracker is not None:
            tracker.append(label, price)
            trend = sparkline(tracker.series(label))
        rows.append((label, prob, price, edge, trend))
    return rows


class LiveTrainingRecorder:
    """Capture live basketball snapshots and append them after the final result."""

    def __init__(self, cache_path: str):
        self.cache_path = Path(cache_path)
        self._rows: list[dict] = []
        self._seen: set[tuple] = set()
        self.saved_rows = 0

    @property
    def buffered_count(self) -> int:
        return len(self._rows)

    def capture(self, event_id: str, detail: LiveDetail) -> None:
        if detail.sport != "basketball":
            return
        state = detail_to_basketball_state(detail)
        key = (
            event_id,
            state.period,
            round(state.seconds_remaining, 3),
            state.home_score,
            state.away_score,
        )
        if key in self._seen:
            return
        self._seen.add(key)
        self._rows.append(
            {
                "game_id": str(event_id),
                "captured_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "home_team": detail.home_team,
                "away_team": detail.away_team,
                "home_score": state.home_score,
                "away_score": state.away_score,
                "period": state.period,
                "seconds_remaining": state.seconds_remaining,
                "pre_game_home_prob": state.pre_game_home_prob,
                "home_recent_net_rating": state.home_recent_net_rating,
                "away_recent_net_rating": state.away_recent_net_rating,
            }
        )

    def save_if_final(self, detail: LiveDetail) -> int:
        if detail.sport != "basketball" or detail.status != "post" or not self._rows:
            return 0
        home_win = int(detail.home_score > detail.away_score)
        df = pd.DataFrame([{**row, "home_win": home_win} for row in self._rows])
        if self.cache_path.exists():
            old = pd.read_parquet(self.cache_path)
            df = pd.concat([old, df], ignore_index=True)
        dedupe_cols = ["game_id", "period", "seconds_remaining", "home_score", "away_score"]
        df = df.drop_duplicates(subset=dedupe_cols, keep="last").reset_index(drop=True)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        before = len(pd.read_parquet(self.cache_path)) if self.cache_path.exists() else 0
        df.to_parquet(self.cache_path, index=False)
        self.saved_rows = max(0, len(df) - before)
        self._rows.clear()
        return self.saved_rows


def settlement_marks(cfg: Config, detail: LiveDetail) -> dict[str, float]:
    """Final settlement marks for the selected game's configured Kalshi contracts."""
    if detail.status != "post":
        return {}
    if detail.sport == "basketball":
        out: dict[str, float] = {}
        if cfg.market.kalshi_ticker:
            out[cfg.market.kalshi_ticker] = 1.0 if detail.home_score > detail.away_score else 0.0
        if cfg.market.kalshi_away_ticker:
            out[cfg.market.kalshi_away_ticker] = 1.0 if detail.away_score > detail.home_score else 0.0
        return out

    s = cfg.soccer
    home_win = detail.home_score > detail.away_score
    draw = detail.home_score == detail.away_score
    away_win = detail.away_score > detail.home_score
    out: dict[str, float] = {}
    if s.kalshi_home_ticker:
        out[s.kalshi_home_ticker] = 1.0 if home_win else 0.0
    if s.kalshi_draw_ticker:
        out[s.kalshi_draw_ticker] = 1.0 if draw else 0.0
    if s.kalshi_away_ticker:
        out[s.kalshi_away_ticker] = 1.0 if away_win else 0.0
    return out


class PaperTradingEngine:
    """Dashboard-local paper signal engine using the same strategy spine as loops."""

    def __init__(self, cfg: Config, ledger_path: str):
        self.cfg = cfg
        self.ledger = PaperLedger(ledger_path)
        self.executor = PaperExecutor(ledger_path=ledger_path)
        self.strategy = Strategy(
            cfg.edge.min_edge,
            cfg.kelly_fraction,
            cfg.max_stake,
            cfg.bankroll,
            cfg.edge.min_price,
            cfg.edge.max_price,
        )
        self.detectors: dict[str, BottomDetector] = {}

    def _detector(self, ticker: str) -> BottomDetector:
        detector = self.detectors.get(ticker)
        if detector is None:
            detector = BottomDetector(
                self.cfg.edge.dip_threshold,
                self.cfg.edge.min_edge,
                self.cfg.edge.rebound_ticks,
            )
            self.detectors[ticker] = detector
        return detector

    def update(
        self,
        detail: LiveDetail,
        readout_rows: list[ReadoutRow],
        event_id: str = "",
    ) -> list[PaperSignalRow]:
        specs = outcome_specs(self.cfg, detail.sport, detail.home_team, detail.away_team)
        exposure = float(self.ledger.summary().get("open_exposure", 0.0))
        rows: list[PaperSignalRow] = []
        for (label, ticker), (_label, model_p, price, edge, _trend) in zip(specs, readout_rows):
            if not ticker:
                rows.append((label, "-", False, "SKIP", "no ticker configured"))
                continue
            if detail.status != "in":
                rows.append((label, ticker, False, "WAIT", "game not live"))
                continue
            if price is None or edge is None:
                rows.append((label, ticker, False, "WAIT", "no market price"))
                continue

            signal = self._detector(ticker).update(price, model_p)
            order = self.strategy.decide(signal)
            if order is None:
                reason = signal.reason or f"edge {edge:+.3f}; waiting for bottom"
                rows.append((label, ticker, signal.is_bottom, "WAIT", reason))
                continue
            if exposure + order.size > self.cfg.bankroll:
                rows.append((label, ticker, True, "SKIP", "bankroll exposure cap"))
                continue
            selected_team = label.removesuffix(" win")
            fill = self.executor.place(
                order,
                ticker,
                metadata={
                    "event_id": event_id,
                    "sport": detail.sport,
                    "league": detail.league,
                    "home_team": detail.home_team,
                    "away_team": detail.away_team,
                    "selected_team": selected_team,
                },
            )
            exposure += fill.size
            rows.append((label, ticker, True, "PAPER BUY", f"{fill.size:.2f}@{fill.price:.3f}"))
        return rows

    def summary(
        self,
        marks: dict[str, float] | None = None,
        settlements: dict[str, float] | None = None,
    ) -> dict[str, float | int]:
        return self.ledger.summary(marks=marks, settlements=settlements)


# ----- rendering (pure: LiveDetail + rows -> renderable) -----
def _dot(active: bool) -> str:
    return "[green]●[/]" if active else ""


def _events_panel(detail: LiveDetail) -> Panel:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="left")
    grid.add_column(justify="center")
    grid.add_column(justify="center")
    grid.add_row("", Text(detail.away_team, style="bold"), Text(detail.home_team, style="bold"))
    if detail.sport == "basketball":
        grid.add_row("Fouls", str(detail.away_fouls), str(detail.home_fouls))
        grid.add_row(
            "Possession", _dot(detail.possession == "away"), _dot(detail.possession == "home")
        )
        if detail.free_throw_active:
            grid.add_row(Text("FREE THROW", style="bold yellow"), "", "")
    else:
        grid.add_row("Yellow", str(detail.away_yellow), str(detail.home_yellow))
        grid.add_row("Red", str(detail.away_red), str(detail.home_red))
        grid.add_row(
            "Possession", _dot(detail.possession == "away"), _dot(detail.possession == "home")
        )
        if detail.set_piece:
            grid.add_row(Text(detail.set_piece.upper(), style="bold yellow"), "", "")
    return Panel(grid, title="Game detail", border_style="blue")


def _game_context_panel(detail: LiveDetail) -> Panel:
    table = Table(show_header=False, box=None, expand=True)
    table.add_column("field", style="dim")
    table.add_column("value")
    table.add_row("league", detail.league)
    table.add_row("status", detail.status)
    if detail.sport == "basketball":
        table.add_row("period", f"Q{detail.period}")
        table.add_row("clock", detail.clock or "-")
        table.add_row(
            "regulation left",
            f"{regulation_seconds_remaining(detail.period, clock_to_seconds(detail.clock)):.0f}s",
        )
        table.add_row("score diff", f"{detail.home_score - detail.away_score:+d} home")
    else:
        table.add_row("minute", f"{detail.minute:.0f}'")
        table.add_row("minutes left", f"{max(0.0, 90.0 - detail.minute):.0f}")
        table.add_row("goal diff", f"{detail.home_score - detail.away_score:+d} home")
    poss = {"home": detail.home_team, "away": detail.away_team}.get(detail.possession, "unknown")
    table.add_row("possession", poss)
    return Panel(table, title="Game context", border_style="blue")


def _readout_panel(rows: list[ReadoutRow]) -> Panel:
    table = Table(box=None, expand=True)
    table.add_column("outcome")
    table.add_column("model P", justify="right")
    table.add_column("price", justify="right")
    table.add_column("edge", justify="right")
    table.add_column("trend")
    for label, prob, price, edge, trend in rows:
        price_str = "—" if price is None else f"{price:.3f}"
        if edge is None:
            edge_str = "—"
        else:
            style = "green" if edge > 0 else "red"
            edge_str = f"[{style}]{edge:+.3f}[/]"
        table.add_row(label, f"{prob:.3f}", price_str, edge_str, f"[cyan]{trend or '—'}[/]")
    return Panel(table, title="Model vs market — price trend (display only)", border_style="magenta")


def _fmt_prob(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _fmt_int(value: int | None) -> str:
    return "-" if value is None else f"{value:,}"


def _market_panel(rows: list[MarketInfoRow]) -> Panel:
    table = Table(box=None, expand=True)
    table.add_column("outcome")
    table.add_column("ticker")
    table.add_column("status")
    table.add_column("bid", justify="right")
    table.add_column("ask", justify="right")
    table.add_column("last", justify="right")
    table.add_column("vol", justify="right")
    table.add_column("24h", justify="right")
    table.add_column("liq", justify="right")
    table.add_column("oi", justify="right")
    if not rows:
        table.add_row("no configured Kalshi tickers", "-", "-", "-", "-", "-", "-", "-", "-", "-")
    for label, ticker, snap in rows:
        if snap is None:
            table.add_row(label, ticker, "-", "-", "-", "-", "-", "-", "-", "-")
            continue
        table.add_row(
            label,
            snap.ticker,
            snap.status or "-",
            _fmt_prob(snap.yes_bid),
            _fmt_prob(snap.yes_ask),
            _fmt_prob(snap.last_price),
            _fmt_int(snap.volume),
            _fmt_int(snap.volume_24h),
            _fmt_int(snap.liquidity),
            _fmt_int(snap.open_interest),
        )
    return Panel(table, title="Kalshi market detail", border_style="magenta")


def _paper_signal_panel(rows: list[PaperSignalRow]) -> Panel:
    table = Table(box=None, expand=True)
    table.add_column("outcome")
    table.add_column("ticker")
    table.add_column("bottom")
    table.add_column("action")
    table.add_column("reason")
    if not rows:
        table.add_row("-", "-", "-", "OFF", "paper trading disabled")
    for label, ticker, is_bottom, action, reason in rows:
        action_style = "green" if action == "PAPER BUY" else "yellow" if action == "WAIT" else "dim"
        table.add_row(
            label,
            ticker,
            "yes" if is_bottom else "no",
            f"[{action_style}]{action}[/]",
            reason,
        )
    return Panel(table, title="Paper signals", border_style="green")


def _paper_pnl_panel(summary: dict[str, float | int] | None) -> Panel:
    summary = summary or {}
    table = Table(show_header=False, box=None, expand=True)
    table.add_column("field", style="dim")
    table.add_column("value", justify="right")
    table.add_row("fills", str(summary.get("fills", 0)))
    table.add_row("open positions", str(summary.get("open_positions", 0)))
    table.add_row("staked", f"{float(summary.get('staked', 0.0)):.2f}")
    table.add_row("open exposure", f"{float(summary.get('open_exposure', 0.0)):.2f}")
    table.add_row("realized PnL", f"{float(summary.get('realized_pnl', 0.0)):+.2f}")
    table.add_row("unrealized PnL", f"{float(summary.get('unrealized_pnl', 0.0)):+.2f}")
    table.add_row("total PnL", f"{float(summary.get('total_pnl', 0.0)):+.2f}")
    return Panel(table, title="Paper ledger PnL", border_style="green")


def render(
    detail: LiveDetail,
    rows: list[ReadoutRow],
    stale: bool,
    updated_at: str,
    market_rows: list[MarketInfoRow] | None = None,
    paper_signal_rows: list[PaperSignalRow] | None = None,
    paper_summary: dict[str, float | int] | None = None,
    training_status: str = "",
) -> Panel:
    status_label = {"pre": "Scheduled", "in": "LIVE", "post": "Final"}.get(
        detail.status, detail.status
    )
    status_style = {"in": "bold green", "post": "bold red"}.get(detail.status, "dim")

    header = Text.assemble(
        (f"{detail.away_team} @ {detail.home_team}", "bold"),
        ("   "),
        (status_label, status_style),
        (f"  {detail.clock}", "dim"),
    )
    score = Text(
        f"{detail.away_team} {detail.away_score}   —   {detail.home_score} {detail.home_team}",
        style="bold cyan",
        justify="center",
    )
    last_play = Panel(Text(detail.last_play_text or "—"), title="Last play", border_style="grey50")
    footer = Text(
        ("[stale — keeping last frame]  " if stale else "") + f"updated {updated_at}",
        style="yellow" if stale else "dim",
    )
    if training_status:
        footer.append(f"  |  {training_status}")
    body = Group(
        header,
        Text(""),
        score,
        Text(""),
        _game_context_panel(detail),
        _events_panel(detail),
        last_play,
        _readout_panel(rows),
        _market_panel(market_rows or []),
        _paper_signal_panel(paper_signal_rows or []),
        _paper_pnl_panel(paper_summary),
        footer,
    )
    return Panel(body, title="SportEdge Live", border_style="cyan")


# ----- picker + live loop (I/O) -----
def pick_game(console: Console, candidates: list[GameCandidate]) -> GameCandidate | None:
    """Show selectable games (live first, then upcoming; finished hidden) and read
    the user's choice. Returns None if nothing to pick or the user quits."""
    selectable = [c for c in candidates if c.status in ("in", "pre")]
    selectable.sort(key=lambda c: 0 if c.status == "in" else 1)
    if not selectable:
        console.print("[yellow]No live or upcoming games found right now.[/]")
        return None

    console.print("[bold]Pick a game:[/]")
    for i, c in enumerate(selectable, 1):
        tag = "[green]LIVE[/]" if c.status == "in" else "[dim]upcoming[/]"
        detail = f" ({c.short_detail})" if c.short_detail else ""
        console.print(f"  {i}. {tag} {c.away_team} @ {c.home_team}  [dim]{c.sport}[/]{detail}")

    choice = console.input("Number (or q to quit): ").strip().lower()
    if choice in ("", "q"):
        return None
    try:
        idx = int(choice) - 1
    except ValueError:
        console.print("[red]Not a number.[/]")
        return None
    if 0 <= idx < len(selectable):
        return selectable[idx]
    console.print("[red]Out of range.[/]")
    return None


def apply_market_coverage(cfg: Config, coverage: GameMarketCoverage) -> None:
    """Apply scanner-discovered markets to the in-memory config for this run.

    Handles basketball (home/away win) and soccer (home/away sides of the 1X2;
    the draw contract stays config-supplied)."""
    game = coverage.game
    if game.sport == "basketball":
        cfg.market.home_team = game.home_team
        cfg.market.away_team = game.away_team
        if coverage.home_market is not None:
            cfg.market.kalshi_ticker = coverage.home_market.ticker
        if coverage.away_market is not None:
            cfg.market.kalshi_away_ticker = coverage.away_market.ticker
    elif game.sport == "soccer":
        cfg.soccer.home_team = game.home_team
        cfg.soccer.away_team = game.away_team
        if coverage.home_market is not None:
            cfg.soccer.kalshi_home_ticker = coverage.home_market.ticker
        if coverage.away_market is not None:
            cfg.soccer.kalshi_away_ticker = coverage.away_market.ticker


def pick_ready_game(
    console: Console,
    candidates: list[GameCandidate],
    cfg: Config,
    client: KalshiClient,
) -> GameCandidate | None:
    """Auto-pick the first live/upcoming game with discovered Kalshi coverage."""
    console.print("[dim]Scanning games for direct quoted Kalshi winner markets...[/]")
    coverage_rows = scan_game_markets(candidates, client)
    for coverage in coverage_rows:
        if coverage.has_market:
            apply_market_coverage(cfg, coverage)
            game = coverage.game
            console.print(
                f"Auto-picked [cyan]{game.away_team} @ {game.home_team}[/] "
                f"with {coverage.market_count} Kalshi market(s)."
            )
            return game
    console.print("[yellow]No ready Kalshi-covered game found.[/]")
    return None


def wait_for_ready_game(
    console: Console,
    cfg: Config,
    client: KalshiClient,
    poll_seconds: int,
    max_wait_seconds: int = 0,
) -> GameCandidate | None:
    """Poll until a Kalshi-covered game appears or max_wait_seconds elapses.

    ``max_wait_seconds=0`` means wait indefinitely; Ctrl-C still exits cleanly.
    """
    started = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        candidate = pick_ready_game(console, list_live_games(), cfg, client)
        if candidate is not None:
            return candidate
        if max_wait_seconds and time.monotonic() - started >= max_wait_seconds:
            console.print(
                f"[yellow]No ready game after {max_wait_seconds}s "
                f"and {attempts} scan(s).[/]"
            )
            return None
        console.print(f"[dim]Waiting {poll_seconds}s before next market scan...[/]")
        time.sleep(poll_seconds)


def load_model(sport: str, path: str):
    """Load the sport's win-prob model, falling back to the built-in
    untrained model if the saved file is missing or can't be unpickled
    (e.g. a trained estimator whose library isn't installed)."""
    factory = WinProbModel if sport == "basketball" else SoccerWinProbModel
    try:
        return factory.load(path)
    except Exception:  # noqa: BLE001 - degrade to the logistic/Poisson fallback
        return factory()


def auto_configure_kalshi_market(
    cfg: Config,
    candidate: GameCandidate,
    client: KalshiClient,
    console: Console,
) -> None:
    """Fill missing Kalshi tickers for this run using conservative discovery.

    Works for both basketball (home/away win) and soccer (home/away win sides of
    the 1X2; the draw contract stays config-supplied). Discovery itself is sport
    agnostic, so the only thing that differs is which config block we populate.
    """
    if candidate.sport == "basketball":
        home_set, away_set = cfg.market.kalshi_ticker, cfg.market.kalshi_away_ticker
    elif candidate.sport == "soccer":
        home_set, away_set = cfg.soccer.kalshi_home_ticker, cfg.soccer.kalshi_away_ticker
    else:
        return

    found = []
    if not home_set:
        result = client.discover_team_win_market(candidate.home_team, candidate.away_team)
        if result is not None:
            if candidate.sport == "basketball":
                cfg.market.kalshi_ticker = result.ticker
            else:
                cfg.soccer.kalshi_home_ticker = result.ticker
            found.append((candidate.home_team, result))
    if not away_set:
        result = client.discover_team_win_market(candidate.away_team, candidate.home_team)
        if result is not None:
            if candidate.sport == "basketball":
                cfg.market.kalshi_away_ticker = result.ticker
            else:
                cfg.soccer.kalshi_away_ticker = result.ticker
            found.append((candidate.away_team, result))
    if not found:
        console.print(
            "[yellow]No direct quoted Kalshi team-win market found; "
            "running model-only/paper-signal disabled for this game.[/]"
        )
        return
    if candidate.sport == "basketball":
        cfg.market.home_team = candidate.home_team
        cfg.market.away_team = candidate.away_team
    else:
        cfg.soccer.home_team = candidate.home_team
        cfg.soccer.away_team = candidate.away_team
    for team, result in found:
        console.print(
            f"Auto-selected Kalshi market: [cyan]{result.ticker}[/] "
            f"for [bold]{team} win[/] "
            f"(ask={result.yes_ask if result.yes_ask is not None else '-'}, "
            f"bid={result.yes_bid if result.yes_bid is not None else '-'})"
        )


def run(
    config_path: str = "config/config.yaml",
    training_cache: str = "data/cache/training.parquet",
    record_training: bool = True,
    paper_ledger: str = "data/cache/paper_ledger.parquet",
    paper_trading: bool = True,
    auto_pick_ready: bool = False,
    wait_ready: bool = False,
    wait_ready_seconds: int = 0,
) -> None:
    from rich.live import Live  # local import keeps module import light for tests

    cfg = load_config(config_path)
    secrets = load_secrets()
    console = Console()

    client = KalshiClient(secrets.kalshi_host, secrets)
    if wait_ready:
        candidate = wait_for_ready_game(
            console,
            cfg,
            client,
            poll_seconds=cfg.loop.poll_seconds,
            max_wait_seconds=wait_ready_seconds,
        )
    else:
        candidates = list_live_games()
        candidate = (
            pick_ready_game(console, candidates, cfg, client)
            if auto_pick_ready
            else pick_game(console, candidates)
        )
    if candidate is None:
        return

    model = load_model(candidate.sport, cfg.model.path)
    auto_configure_kalshi_market(cfg, candidate, client, console)
    recorder = LiveTrainingRecorder(training_cache) if record_training else None
    paper_engine = PaperTradingEngine(cfg, paper_ledger) if paper_trading else None

    # Seed the price-trend tracker from Kalshi candlesticks (historical chart);
    # the loop then extends each series with live samples (sparkline fallback).
    tracker = TrendTracker()
    for label, ticker in outcome_specs(
        cfg, candidate.sport, candidate.home_team, candidate.away_team
    ):
        if ticker:
            tracker.seed(label, client.get_candlesticks(ticker.split("-")[0], ticker))

    console.print(
        f"Watching [cyan]{candidate.away_team} @ {candidate.home_team}[/] "
        f"({candidate.sport}). Ctrl-C to stop."
    )
    if recorder and candidate.sport == "basketball":
        console.print(f"Training capture: [cyan]{training_cache}[/] (saved when final)")
    elif recorder:
        console.print("[dim]Training capture: soccer snapshots are display-only for now.[/]")
    if paper_engine:
        console.print(f"Paper trading: [cyan]{paper_ledger}[/] (no live orders)")

    last: LiveDetail | None = None
    try:
        with Live(console=console, refresh_per_second=4, screen=False) as live:
            while True:
                detail = get_game_detail(candidate.sport, candidate.league, candidate.event_id)
                stale = detail is None
                if detail is not None:
                    last = detail
                if last is not None:
                    if recorder and detail is not None:
                        recorder.capture(candidate.event_id, detail)
                    rows = build_readout(cfg, last, model, client, tracker)
                    market_rows = build_market_info(cfg, last, client)
                    marks = {
                        ticker: row[2]
                        for (_label, ticker), row in zip(
                            outcome_specs(cfg, last.sport, last.home_team, last.away_team), rows
                        )
                        if ticker and row[2] is not None
                    }
                    settlements = settlement_marks(cfg, last)
                    paper_signal_rows: list[PaperSignalRow] = []
                    paper_summary: dict[str, float | int] | None = None
                    if paper_engine:
                        paper_signal_rows = paper_engine.update(last, rows, candidate.event_id)
                        paper_summary = paper_engine.summary(
                            marks=marks,
                            settlements=settlements,
                        )
                    training_status = ""
                    if recorder and last.sport == "basketball":
                        training_status = f"training rows buffered={recorder.buffered_count}"
                    live.update(
                        render(
                            last,
                            rows,
                            stale,
                            datetime.now().strftime("%H:%M:%S"),
                            market_rows=market_rows,
                            paper_signal_rows=paper_signal_rows,
                            paper_summary=paper_summary,
                            training_status=training_status,
                        )
                    )
                    if last.status == "post":
                        if recorder:
                            saved = recorder.save_if_final(last)
                            if saved:
                                console.print(
                                    f"[green]Saved {saved} labeled training rows -> {training_cache}[/]"
                                )
                        break
                time.sleep(cfg.loop.poll_seconds)
    except KeyboardInterrupt:
        pass
    console.print("[bold]Dashboard closed.[/]")


def main() -> None:
    ap = argparse.ArgumentParser(description="SportEdge live game dashboard")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--training-cache", default="data/cache/training.parquet")
    ap.add_argument("--no-record-training", action="store_true")
    ap.add_argument("--paper-ledger", default="data/cache/paper_ledger.parquet")
    ap.add_argument("--no-paper-trading", action="store_true")
    ap.add_argument("--auto-pick-ready", action="store_true")
    ap.add_argument("--wait-ready", action="store_true")
    ap.add_argument(
        "--wait-ready-seconds",
        type=int,
        default=0,
        help="Max seconds to wait for a Kalshi-covered game; 0 waits forever",
    )
    args = ap.parse_args()
    run(
        config_path=args.config,
        training_cache=args.training_cache,
        record_training=not args.no_record_training,
        paper_ledger=args.paper_ledger,
        paper_trading=not args.no_paper_trading,
        auto_pick_ready=args.auto_pick_ready,
        wait_ready=args.wait_ready,
        wait_ready_seconds=args.wait_ready_seconds,
    )


if __name__ == "__main__":
    main()
