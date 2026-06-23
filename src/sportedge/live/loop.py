"""Live orchestration loop: game state -> model P(win) -> market price ->
edge + bottom check -> strategy -> executor. Paper mode by default.

    python -m sportedge.live.loop --mode paper
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.table import Table

from sportedge.betting.executor import make_executor, paper_gate_status
from sportedge.betting.flow import detect_flow
from sportedge.betting.risk import RiskManager, state_from_fills
from sportedge.betting.strategy import Strategy
from sportedge.config import load_config, load_secrets
from sportedge.data.isports import get_live_state as get_isports_live_state
from sportedge.data.nba_scraper import get_live_state
from sportedge.market.edge import BottomDetector, edge
from sportedge.market.kalshi import KalshiClient
from sportedge.model.live_winprob import WinProbModel

console = Console()


def _live_state(home_team: str, away_team: str):
    state = get_live_state(home_team, away_team, 0.60)
    if state is not None:
        return state
    return get_isports_live_state(home_team, away_team, 0.60)


def _auto_discover_ticker(cfg, client: KalshiClient) -> str | None:
    if cfg.market.kalshi_ticker:
        return cfg.market.kalshi_ticker
    if not cfg.market.home_team:
        return None
    result = client.discover_team_win_market(cfg.market.home_team, cfg.market.away_team)
    if result is None:
        return None
    cfg.market.kalshi_ticker = result.ticker
    console.print(
        f"Auto-selected Kalshi market: [cyan]{result.ticker}[/] "
        f"for [bold]{cfg.market.home_team} win[/]"
    )
    return result.ticker


def paper_metadata(cfg) -> dict[str, str]:
    """Metadata required for later ESPN settlement of NBA paper fills."""
    return {
        "event_id": cfg.market.espn_event_id or cfg.market.market_slug,
        "sport": "basketball",
        "league": "nba",
        "home_team": cfg.market.home_team,
        "away_team": cfg.market.away_team,
        "selected_team": cfg.market.home_team,
    }


def run(
    mode: str | None = None,
    config_path: str = "config/config.yaml",
    paper_ledger: str = "data/cache/paper_ledger.parquet",
) -> None:
    cfg = load_config(config_path)
    if mode:
        cfg.mode = mode
    secrets = load_secrets()

    model = WinProbModel.load(cfg.model.path)
    client = KalshiClient(secrets.kalshi_host, secrets)
    strategy = Strategy(
        cfg.edge.min_edge,
        cfg.kelly_fraction,
        cfg.max_stake,
        cfg.bankroll,
        cfg.edge.min_price,
        cfg.edge.max_price,
    )
    detector = BottomDetector(cfg.edge.dip_threshold, cfg.edge.min_edge, cfg.edge.rebound_ticks)
    executor = make_executor(cfg, secrets, paper_ledger_path=paper_ledger)
    risk_manager = RiskManager(cfg.risk)
    proof_ok, proof_reason = paper_gate_status(cfg, paper_ledger)

    console.rule(
        f"[bold]SportEdge live loop[/] - venue=[bold]kalshi[/] mode=[bold]{executor.mode}[/] "
        f"model={'trained' if model.is_trained else 'logistic-fallback'}"
    )
    if executor.mode == "live":
        console.print("[red bold]LIVE MODE: real orders will be placed.[/]")
    else:
        console.print(f"Paper ledger: [cyan]{paper_ledger}[/]")
        if cfg.live_enabled and secrets.kalshi_complete and not proof_ok:
            console.print(f"[yellow]Live blocked by paper gate: {proof_reason}[/]")

    # The home-team "win" contract is a single configured Kalshi ticker.
    token_id: str | None = _auto_discover_ticker(cfg, client)
    if token_id:
        console.print(f"Market: [cyan]{token_id}[/] (Kalshi home-win contract)")
    else:
        console.print("[yellow]No Kalshi ticker configured; running model-only.[/]")

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
                price = client.get_price(token_id, "BUY")
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]price fetch failed: {exc}[/]")

        placed = ""
        blocked = ""
        if price is not None:
            sig = detector.update(price, model_p)
            order = strategy.decide(sig)
            if order:
                meta = paper_metadata(cfg)
                decision = risk_manager.check(
                    order_size=order.size,
                    token_id=token_id or "",
                    event_id=meta["event_id"],
                    state=state_from_fills(executor.fills, cfg.bankroll),
                )
                if not decision.allowed:
                    blocked = decision.reason
                else:
                    flow_ok = True
                    if cfg.flow.mode == "confirm" and token_id:
                        flow = detect_flow(client.get_trades(token_id), cfg.flow)
                        flow_ok = flow.confirms_buy
                        if not flow_ok:
                            blocked = f"flow: {flow.reason}"
                    if flow_ok:
                        fill = executor.place(order, token_id or "", metadata=meta)
                        placed = (
                            f"BUY {fill.size:.2f}@{fill.price:.3f}"
                            + (f" [{fill.status}]" if fill.status not in ("", "paper") else "")
                        )

        _render(state, model_p, price, executor, placed, blocked)

        if state.is_final:
            console.rule("[bold]Game final[/]")
            break
        time.sleep(cfg.loop.poll_seconds)


def _render(state, model_p, price, executor, placed: str, blocked: str = "") -> None:
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
    if blocked:
        t.add_row("[yellow]risk block[/]", blocked)
    console.print(t)


def main() -> None:
    ap = argparse.ArgumentParser(description="SportEdge live loop")
    ap.add_argument("--mode", choices=["paper", "live"], default=None)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--paper-ledger", default="data/cache/paper_ledger.parquet")
    args = ap.parse_args()
    run(mode=args.mode, config_path=args.config, paper_ledger=args.paper_ledger)


if __name__ == "__main__":
    main()
