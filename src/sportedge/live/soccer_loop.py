"""Live World Cup orchestration loop (3-way 1X2).

Per tick: soccer game state -> model P(home)/P(draw)/P(away) -> three Kalshi
prices -> per-outcome edge + bottom check -> Kelly sizing -> executor. Each of the
three outcome contracts gets its own ``BottomDetector`` so a dip on any one can be
sniped independently. Paper mode by default.

    python -m sportedge.live.soccer_loop --mode paper
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.table import Table

from sportedge.betting.executor import make_executor, paper_gate_status
from sportedge.betting.strategy import Strategy
from sportedge.config import load_config, load_secrets
from sportedge.data.espn_soccer import get_live_state
from sportedge.market.edge import BottomDetector, edge
from sportedge.market.kalshi import KalshiClient
from sportedge.model.soccer_winprob import SoccerWinProbModel
from sportedge.types import SoccerGameState, WinProb3

console = Console()

# Logical outcome keys in display order.
OUTCOMES = ("home", "draw", "away")


def _model_p(probs: WinProb3, key: str) -> float:
    return {"home": probs.home, "draw": probs.draw, "away": probs.away}[key]


def _live_state(cfg) -> SoccerGameState | None:
    s = cfg.soccer
    return get_live_state(s.home_team, s.away_team, s.lambda_home, s.lambda_away, s.league)


def _market_client(cfg, secrets) -> KalshiClient:
    """Kalshi market client. Exposes ``get_price(ticker, side) -> [0, 1]``."""
    return KalshiClient(secrets.kalshi_host, secrets)


def _resolve_tokens(cfg, client) -> dict[str, str]:
    """Map each 1X2 outcome to its configured Kalshi contract ticker."""
    s = cfg.soccer
    tickers = {
        "home": s.kalshi_home_ticker,
        "draw": s.kalshi_draw_ticker,
        "away": s.kalshi_away_ticker,
    }
    return {key: t for key, t in tickers.items() if t}


def paper_metadata(cfg, outcome: str) -> dict[str, str]:
    """Metadata required for later ESPN settlement of soccer paper fills."""
    selected = {
        "home": cfg.soccer.home_team,
        "draw": "draw",
        "away": cfg.soccer.away_team,
    }[outcome]
    return {
        "event_id": cfg.soccer.espn_event_id or cfg.soccer.market_slug,
        "sport": "soccer",
        "league": cfg.soccer.league,
        "home_team": cfg.soccer.home_team,
        "away_team": cfg.soccer.away_team,
        "selected_team": selected,
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

    model = SoccerWinProbModel.load(cfg.soccer.model_path)
    client = _market_client(cfg, secrets)
    strategy = Strategy(
        cfg.edge.min_edge,
        cfg.kelly_fraction,
        cfg.max_stake,
        cfg.bankroll,
        cfg.edge.min_price,
        cfg.edge.max_price,
    )
    detectors = {
        key: BottomDetector(cfg.edge.dip_threshold, cfg.edge.min_edge, cfg.edge.rebound_ticks)
        for key in OUTCOMES
    }
    executor = make_executor(cfg, secrets, paper_ledger_path=paper_ledger)
    proof_ok, proof_reason = paper_gate_status(cfg, paper_ledger)

    console.rule(
        f"[bold]SportEdge WC loop[/] - {cfg.soccer.home_team} vs {cfg.soccer.away_team} "
        f"- venue=[bold]kalshi[/] mode=[bold]{executor.mode}[/] "
        f"model={'trained' if model.is_trained else 'poisson-fallback'}"
    )
    if executor.mode == "live":
        console.print("[red bold]LIVE MODE: real orders will be placed.[/]")
    else:
        console.print(f"Paper ledger: [cyan]{paper_ledger}[/]")
        if cfg.live_enabled and secrets.kalshi_complete and not proof_ok:
            console.print(f"[yellow]Live blocked by paper gate: {proof_reason}[/]")

    # Resolve the three outcome tokens once.
    tokens: dict[str, str] = {}
    try:
        tokens = _resolve_tokens(cfg, client)
        if tokens:
            console.print(
                f"Venue [cyan]kalshi[/]  tickers={ {k: v[:8] for k, v in tokens.items()} }"
            )
        else:
            console.print("[yellow]No outcome tokens resolved; running model-only.[/]")
    except Exception as exc:  # noqa: BLE001 - degrade to model-only display
        console.print(f"[yellow]Market lookup failed ({exc}); running model-only.[/]")

    while True:
        try:
            state = _live_state(cfg)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Live state lookup failed: {exc}[/]")
            state = None

        if state is None:
            console.print("[yellow]No matching live match yet; waiting...[/]")
            time.sleep(cfg.loop.poll_seconds)
            continue

        probs = model.predict(state)

        prices: dict[str, float] = {}
        placed: list[str] = []
        for key in OUTCOMES:
            token_id = tokens.get(key)
            if not token_id:
                continue
            try:
                price = client.get_price(token_id, "BUY")
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]{key} price fetch failed: {exc}[/]")
                continue
            if price is None:
                continue
            prices[key] = price

            sig = detectors[key].update(price, _model_p(probs, key))
            order = strategy.decide(sig)
            if order and executor.staked + order.size <= cfg.bankroll:
                fill = executor.place(
                    order,
                    token_id,
                    metadata=paper_metadata(cfg, key),
                )
                placed.append(f"{key.upper()} {fill.size:.2f}@{fill.price:.3f}")

        _render(state, probs, prices, executor, placed)

        if state.is_final:
            console.rule("[bold]Match final[/]")
            break
        time.sleep(cfg.loop.poll_seconds)


def _render(
    state: SoccerGameState,
    probs: WinProb3,
    prices: dict[str, float],
    executor,
    placed: list[str],
) -> None:
    t = Table(show_header=True, box=None)
    t.add_column("outcome")
    t.add_column("model P")
    t.add_column("price")
    t.add_column("edge")

    model_by_key = {"home": probs.home, "draw": probs.draw, "away": probs.away}
    label_by_key = {
        "home": state.home_team,
        "draw": "Draw",
        "away": state.away_team,
    }
    for key in OUTCOMES:
        mp = model_by_key[key]
        price = prices.get(key)
        edge_str = f"{edge(mp, price):+.3f}" if price is not None else "-"
        price_str = f"{price:.3f}" if price is not None else "-"
        t.add_row(label_by_key[key], f"{mp:.3f}", price_str, edge_str)

    console.print(
        f"[bold]{state.home_team} {state.home_goals}-{state.away_goals} {state.away_team}[/]  "
        f"{state.minute:.0f}'  (reds {state.home_red_cards}-{state.away_red_cards})"
    )
    console.print(t)
    console.print(f"staked {executor.staked:.2f} ({len(executor.fills)} fills)")
    if placed:
        console.print("[green]ORDERS:[/] " + ", ".join(placed))


def main() -> None:
    ap = argparse.ArgumentParser(description="SportEdge World Cup live loop")
    ap.add_argument("--mode", choices=["paper", "live"], default=None)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--paper-ledger", default="data/cache/paper_ledger.parquet")
    args = ap.parse_args()
    run(mode=args.mode, config_path=args.config, paper_ledger=args.paper_ledger)


if __name__ == "__main__":
    main()
