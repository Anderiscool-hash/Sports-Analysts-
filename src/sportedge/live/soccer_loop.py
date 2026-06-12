"""Live World Cup orchestration loop (3-way 1X2).

Per tick: soccer game state -> model P(home)/P(draw)/P(away) -> three Polymarket
prices -> per-outcome edge + bottom check -> Kelly sizing -> executor. Each of the
three outcome tokens gets its own ``BottomDetector`` so a dip on any one can be
sniped independently. Paper mode by default.

    python -m sportedge.live.soccer_loop --mode paper
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.table import Table

from sportedge.betting.executor import make_executor
from sportedge.betting.strategy import Strategy
from sportedge.config import load_config, load_secrets
from sportedge.data.espn_soccer import get_live_state
from sportedge.market.edge import BottomDetector, edge
from sportedge.market.kalshi import KalshiClient
from sportedge.market.polymarket import PolymarketClient
from sportedge.model.soccer_winprob import SoccerWinProbModel
from sportedge.types import SoccerGameState, WinProb3

console = Console()

# Logical outcome keys in display order.
OUTCOMES = ("home", "draw", "away")


def map_outcome_tokens(
    outcomes: list[str],
    token_ids: list[str],
    home_label: str,
    draw_label: str,
    away_label: str,
) -> dict[str, str]:
    """Match each 1X2 outcome to its CLOB token id by label substring.

    Returns a dict with any subset of {"home","draw","away"} that could be resolved.
    """
    wanted = {"home": home_label, "draw": draw_label, "away": away_label}
    result: dict[str, str] = {}
    for key, label in wanted.items():
        if not label:
            continue
        label_l = label.lower()
        for i, outcome in enumerate(outcomes):
            text = str(outcome).lower()
            if i < len(token_ids) and (label_l in text or text in label_l):
                result[key] = token_ids[i]
                break
    return result


def _model_p(probs: WinProb3, key: str) -> float:
    return {"home": probs.home, "draw": probs.draw, "away": probs.away}[key]


def _live_state(cfg) -> SoccerGameState | None:
    s = cfg.soccer
    return get_live_state(s.home_team, s.away_team, s.lambda_home, s.lambda_away, s.league)


def _market_client(cfg, secrets):
    """Pick the market venue. Both clients expose ``get_price(token, side) -> [0,1]``."""
    if cfg.venue == "kalshi":
        return KalshiClient(secrets.kalshi_host, secrets)
    return PolymarketClient(cfg.soccer.gamma_host, secrets.clob_host, secrets.chain_id, secrets)


def _resolve_tokens(cfg, client) -> dict[str, str]:
    """Map each 1X2 outcome to a venue token id. Kalshi uses configured contract
    tickers; Polymarket discovers the market and matches outcome labels."""
    s = cfg.soccer
    if cfg.venue == "kalshi":
        tickers = {
            "home": s.kalshi_home_ticker,
            "draw": s.kalshi_draw_ticker,
            "away": s.kalshi_away_ticker,
        }
        return {key: t for key, t in tickers.items() if t}
    market = client.find_market(s.market_slug, query=f"{s.home_team} {s.away_team}")
    if market and market.token_ids:
        return map_outcome_tokens(
            market.outcomes,
            market.token_ids,
            s.home_outcome or s.home_team,
            s.draw_outcome,
            s.away_outcome or s.away_team,
        )
    return {}


def run(mode: str | None = None, config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)
    if mode:
        cfg.mode = mode
    secrets = load_secrets()

    model = SoccerWinProbModel.load(cfg.model.path)
    client = _market_client(cfg, secrets)
    strategy = Strategy(cfg.edge.min_edge, cfg.kelly_fraction, cfg.max_stake, cfg.bankroll)
    detectors = {
        key: BottomDetector(cfg.edge.dip_threshold, cfg.edge.min_edge, cfg.edge.rebound_ticks)
        for key in OUTCOMES
    }
    executor = make_executor(cfg, secrets)

    console.rule(
        f"[bold]SportEdge WC loop[/] - {cfg.soccer.home_team} vs {cfg.soccer.away_team} "
        f"- venue=[bold]{cfg.venue}[/] mode=[bold]{executor.mode}[/] "
        f"model={'trained' if model.is_trained else 'poisson-fallback'}"
    )
    if executor.mode == "live":
        console.print("[red bold]LIVE MODE: real orders will be placed.[/]")

    # Resolve the three outcome tokens once.
    tokens: dict[str, str] = {}
    try:
        tokens = _resolve_tokens(cfg, client)
        if tokens:
            console.print(
                f"Venue [cyan]{cfg.venue}[/]  tokens={ {k: v[:8] for k, v in tokens.items()} }"
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
                fill = executor.place(order, token_id)
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
    args = ap.parse_args()
    run(mode=args.mode, config_path=args.config)


if __name__ == "__main__":
    main()
