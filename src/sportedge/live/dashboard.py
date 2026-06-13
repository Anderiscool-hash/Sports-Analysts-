"""Live game dashboard — a rich, in-terminal watch screen.

Pick a live (or upcoming) game, then watch a constantly-updating panel: score,
clock, sport-specific detail (basketball fouls / free throws; soccer cards /
set-pieces), possession, the raw last play, and your model's win-probability with
a best-effort Kalshi price + edge alongside.

Display-only: it never places an order.

    python -m sportedge.live.dashboard
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sportedge.config import Config, load_config, load_secrets
from sportedge.data.espn_live import get_game_detail, list_live_games
from sportedge.market.kalshi import KalshiClient
from sportedge.model.live_winprob import WinProbModel
from sportedge.model.soccer_winprob import SoccerWinProbModel
from sportedge.types import (
    GameCandidate,
    GameState,
    LiveDetail,
    SoccerGameState,
    regulation_seconds_remaining,
)

# One readout row: outcome label, model P, market price (or None), edge (or None).
ReadoutRow = tuple[str, float, float | None, float | None]


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


def build_readout(
    cfg: Config, detail: LiveDetail, model, client: KalshiClient | None
) -> list[ReadoutRow]:
    """Model win-prob + (best-effort) Kalshi price + edge per outcome."""
    rows: list[ReadoutRow] = []
    if detail.sport == "basketball":
        prob = model.predict(detail_to_basketball_state(detail))
        price = _safe_price(client, cfg.market.kalshi_ticker)
        edge = (prob - price) if price is not None else None
        rows.append((f"{detail.home_team} win", prob, price, edge))
    else:
        probs = model.predict(
            detail_to_soccer_state(detail, cfg.soccer.lambda_home, cfg.soccer.lambda_away)
        )
        s = cfg.soccer
        spec = [
            (detail.home_team, probs.home, s.kalshi_home_ticker),
            ("Draw", probs.draw, s.kalshi_draw_ticker),
            (detail.away_team, probs.away, s.kalshi_away_ticker),
        ]
        for label, prob, ticker in spec:
            price = _safe_price(client, ticker)
            edge = (prob - price) if price is not None else None
            rows.append((label, prob, price, edge))
    return rows


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


def _readout_panel(rows: list[ReadoutRow]) -> Panel:
    table = Table(box=None, expand=True)
    table.add_column("outcome")
    table.add_column("model P", justify="right")
    table.add_column("price", justify="right")
    table.add_column("edge", justify="right")
    for label, prob, price, edge in rows:
        price_str = "—" if price is None else f"{price:.3f}"
        if edge is None:
            edge_str = "—"
        else:
            style = "green" if edge > 0 else "red"
            edge_str = f"[{style}]{edge:+.3f}[/]"
        table.add_row(label, f"{prob:.3f}", price_str, edge_str)
    return Panel(table, title="Model vs market (display only)", border_style="magenta")


def render(detail: LiveDetail, rows: list[ReadoutRow], stale: bool, updated_at: str) -> Panel:
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
    body = Group(
        header,
        Text(""),
        score,
        Text(""),
        _events_panel(detail),
        last_play,
        _readout_panel(rows),
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


def load_model(sport: str, path: str):
    """Load the sport's win-prob model, falling back to the built-in
    untrained model if the saved file is missing or can't be unpickled
    (e.g. a trained estimator whose library isn't installed)."""
    factory = WinProbModel if sport == "basketball" else SoccerWinProbModel
    try:
        return factory.load(path)
    except Exception:  # noqa: BLE001 - degrade to the logistic/Poisson fallback
        return factory()


def run(config_path: str = "config/config.yaml") -> None:
    from rich.live import Live  # local import keeps module import light for tests

    cfg = load_config(config_path)
    secrets = load_secrets()
    console = Console()

    candidate = pick_game(console, list_live_games())
    if candidate is None:
        return

    model = load_model(candidate.sport, cfg.model.path)
    client = KalshiClient(secrets.kalshi_host, secrets)

    console.print(
        f"Watching [cyan]{candidate.away_team} @ {candidate.home_team}[/] "
        f"({candidate.sport}). Ctrl-C to stop."
    )

    last: LiveDetail | None = None
    try:
        with Live(console=console, refresh_per_second=4, screen=False) as live:
            while True:
                detail = get_game_detail(candidate.sport, candidate.league, candidate.event_id)
                stale = detail is None
                if detail is not None:
                    last = detail
                if last is not None:
                    rows = build_readout(cfg, last, model, client)
                    live.update(render(last, rows, stale, datetime.now().strftime("%H:%M:%S")))
                    if last.status == "post":
                        break
                time.sleep(cfg.loop.poll_seconds)
    except KeyboardInterrupt:
        pass
    console.print("[bold]Dashboard closed.[/]")


def main() -> None:
    ap = argparse.ArgumentParser(description="SportEdge live game dashboard")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    run(config_path=args.config)


if __name__ == "__main__":
    main()
