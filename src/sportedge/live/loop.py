"""Live orchestration loop: game state -> model P(win) -> market price ->
edge + bottom check -> strategy -> executor. Paper mode by default.

    python -m sportedge.live.loop --mode paper
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.table import Table

from sportedge.betting.executor import make_executor
from sportedge.betting.strategy import Strategy
from sportedge.config import load_config, load_secrets
from sportedge.data.isports import get_live_state as get_isports_live_state
from sportedge.data.nba_scraper import get_live_state
from sportedge.market.edge import BottomDetector, edge
from sportedge.market.polymarket import PolymarketClient
from sportedge.model.live_winprob import WinProbModel

console = Console()


def _home_token_index(outcomes: list[str], home_team: str) -> int:
    """Pick the CLOB token whose outcome refers to the home team; default 0."""
    home = home_team.lower()
    for i, outcome in enumerate(outcomes):
        if home and home in str(outcome).lower():
            return i
    return 0


def _live_state(home_team: str, away_team: str):
    state = get_live_state(home_team, away_team, 0.60)
    if state is not None:
        return state
    return get_isports_live_state(home_team, away_team, 0.60)


def run(mode: str | None = None, config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)
    if mode:
        cfg.mode = mode
    secrets = load_secrets()

    model = WinProbModel.load(cfg.model.path)
    pm = PolymarketClient(cfg.market.gamma_host, secrets.clob_host, secrets.chain_id, secrets)
    strategy = Strategy(cfg.edge.min_edge, cfg.kelly_fraction, cfg.max_stake, cfg.bankroll)
    detector = BottomDetector(cfg.edge.dip_threshold, cfg.edge.min_edge, cfg.edge.rebound_ticks)
    executor = make_executor(cfg, secrets)

    console.rule(
        f"[bold]SportEdge live loop[/] - mode=[bold]{executor.mode}[/] "
        f"model={'trained' if model.is_trained else 'logistic-fallback'}"
    )
    if executor.mode == "live":
        console.print("[red bold]LIVE MODE: real orders will be placed.[/]")

    # Resolve the Polymarket market + the home-team token once.
    token_id: str | None = None
    try:
        market = pm.find_market(
            cfg.market.market_slug,
            query=f"{cfg.market.home_team} {cfg.market.away_team}",
        )
        if market and market.token_ids:
            idx = _home_token_index(market.outcomes, cfg.market.home_team)
            token_id = market.token_ids[min(idx, len(market.token_ids) - 1)]
            console.print(f"Market: [cyan]{market.question or market.slug}[/]  token={token_id}")
    except Exception as exc:  # noqa: BLE001 - degrade to model-only display
        console.print(f"[yellow]Market lookup failed ({exc}); running model-only.[/]")

    while True:
        try:
            state = _live_state(cfg.market.home_team, cfg.market.away_team)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Live state lookup failed: {exc}[/]")
            state = None

        if state is None:
            console.print("[yellow]No matching live game yet; waiting...[/]")
            time.sleep(cfg.loop.poll_seconds)
            continue

        model_p = model.predict(state)

        price = None
        if token_id:
            try:
                price = pm.get_price(token_id, "BUY")
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]price fetch failed: {exc}[/]")

        placed = ""
        if price is not None:
            sig = detector.update(price, model_p)
            order = strategy.decide(sig)
            if order and executor.staked + order.size <= cfg.bankroll:
                fill = executor.place(order, token_id or "")
                placed = f"BUY {fill.size:.2f}@{fill.price:.3f}"

        _render(state, model_p, price, executor, placed)

        if state.is_final:
            console.rule("[bold]Game final[/]")
            break
        time.sleep(cfg.loop.poll_seconds)


def _render(state, model_p, price, executor, placed: str) -> None:
    t = Table(show_header=False, box=None)
    t.add_row("score", f"{state.home_team} {state.home_score}-{state.away_score} {state.away_team}")
    t.add_row("clock", f"Q{state.period}  {state.seconds_remaining:.0f}s left (reg)")
    t.add_row("model P(home)", f"{model_p:.3f}")
    if price is not None:
        t.add_row("market price", f"{price:.3f}")
        t.add_row("edge", f"{edge(model_p, price):+.3f}")
    t.add_row("staked", f"{executor.staked:.2f} ({len(executor.fills)} fills)")
    if placed:
        t.add_row("[green]ORDER[/]", placed)
    console.print(t)


def main() -> None:
    ap = argparse.ArgumentParser(description="SportEdge live loop")
    ap.add_argument("--mode", choices=["paper", "live"], default=None)
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    run(mode=args.mode, config_path=args.config)


if __name__ == "__main__":
    main()
