"""Discover/fetch Polymarket prices for NBA Finals games and aggregate P&L."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_market_backtest import KAGGLE_ROOT, build_model_states  # noqa: E402
from sportedge.config import load_secrets  # noqa: E402
from sportedge.data.storage import load_parquet, save_parquet  # noqa: E402
from sportedge.market.pnl import align_model_prices, simulate_buy_and_hold, trades_frame  # noqa: E402
from sportedge.market.polymarket import PolymarketClient  # noqa: E402

TEAM_QUERY_NAMES = {
    "Boston Celtics": "Celtics",
    "Dallas Mavericks": "Mavericks",
}


def _json_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(json.loads(value))


def _games(feature_cache: str, season_prefix: str = "00423") -> pd.DataFrame:
    rows = load_parquet(feature_cache)
    if rows is None or rows.empty:
        raise SystemExit(f"No rows in {feature_cache}")
    game_rows = (
        rows[rows["game_id"].str.startswith(season_prefix)]
        .groupby("game_id", as_index=False)
        .agg(home_win=("home_win", "first"))
    )
    # Hard-code the 2024 Finals metadata we can discover reliably from Polymarket.
    meta = pd.DataFrame(
        [
            {
                "game_id": "0042300401",
                "date": "2024-06-06",
                "home": "Boston Celtics",
                "away": "Dallas Mavericks",
            },
            {
                "game_id": "0042300402",
                "date": "2024-06-09",
                "home": "Boston Celtics",
                "away": "Dallas Mavericks",
            },
            {
                "game_id": "0042300403",
                "date": "2024-06-12",
                "home": "Dallas Mavericks",
                "away": "Boston Celtics",
            },
            {
                "game_id": "0042300404",
                "date": "2024-06-14",
                "home": "Dallas Mavericks",
                "away": "Boston Celtics",
            },
            {
                "game_id": "0042300405",
                "date": "2024-06-17",
                "home": "Boston Celtics",
                "away": "Dallas Mavericks",
            },
        ]
    )
    return meta.merge(game_rows, on="game_id", how="inner")


def _team_query_name(name: str) -> str:
    return TEAM_QUERY_NAMES.get(name, name.split()[-1])


def _is_moneyline(question: str, home: str, away: str) -> bool:
    q = question.lower()
    if any(
        term in q
        for term in ("1h", "spread", "o/u", "over", "under", "points", "rebounds", "assists")
    ):
        return False
    return _team_query_name(home).lower() in q and _team_query_name(away).lower() in q


def _game_number(game_id: str) -> int:
    return int(str(game_id)[-1])


def _market_matches_date(market: dict, game_date: str) -> bool:
    text = " ".join(
        str(value or "")
        for value in (market.get("slug"), market.get("endDate"), market.get("endDateIso"))
    )
    return game_date in text


def discover_moneyline(client: PolymarketClient, game: pd.Series, limit: int = 10) -> dict | None:
    home = _team_query_name(game.home)
    away = _team_query_name(game.away)
    queries = [
        f"NBA Finals Game {_game_number(game.game_id)} {home} {away}",
        f"NBA Finals Game {_game_number(game.game_id)} {away} {home}",
        f"{home} {away}",
        f"{away} {home}",
    ]
    seen: set[str] = set()
    for query in queries:
        if query in seen:
            continue
        seen.add(query)
        results = client.public_search(query, limit=limit)
        for event in results.get("events") or []:
            for market in event.get("markets") or []:
                if not _market_matches_date(market, str(game.date)):
                    continue
                if _is_moneyline(str(market.get("question", "")), game.home, game.away):
                    outcomes = _json_list(market.get("outcomes"))
                    tokens = _json_list(market.get("clobTokenIds"))
                    mapping = {
                        str(outcome): str(token)
                        for outcome, token in zip(outcomes, tokens, strict=False)
                    }
                    return {
                        "market_slug": market.get("slug", ""),
                        "question": market.get("question", ""),
                        "home_token": mapping.get(home, ""),
                        "away_token": mapping.get(away, ""),
                    }
    return None


def _price_cache_name(game_id: str, outcome: str) -> str:
    return f"polymarket_{game_id}_{outcome}_10m"


def _fetch_prices(
    client: PolymarketClient,
    token_id: str,
    game_date: str,
    cache_name: str,
    refresh: bool,
) -> pd.DataFrame:
    existing = load_parquet(cache_name)
    if existing is not None and not existing.empty and not refresh:
        return existing
    start = int((pd.Timestamp(game_date, tz="UTC") - timedelta(hours=8)).timestamp())
    end = int((pd.Timestamp(game_date, tz="UTC") + timedelta(days=2)).timestamp())
    df = client.prices_history_frame(
        token_id,
        start_ts=start,
        end_ts=end,
        interval="1m",
        fidelity=10,
    )
    if not df.empty:
        save_parquet(df, cache_name)
    return df


def _parse_edges(values: list[str]) -> list[float]:
    edges: list[float] = []
    for value in values:
        for part in value.split(","):
            text = part.strip()
            if text:
                edges.append(float(text))
    return sorted(set(edges))


def _run_aligned_backtest(
    *,
    aligned: pd.DataFrame,
    token_won: bool,
    min_edge: float,
    cooldown_seconds: int,
) -> pd.DataFrame:
    trades = simulate_buy_and_hold(
        aligned,
        token_won=token_won,
        min_edge=min_edge,
        cooldown_seconds=cooldown_seconds,
    )
    return trades_frame(trades)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run batch Finals Polymarket P&L backtest")
    ap.add_argument("--feature-cache", default="training_kaggle_2021_24_with_team_form")
    ap.add_argument("--model-path", default="models/winprob.joblib")
    ap.add_argument("--kaggle-root", default=str(KAGGLE_ROOT))
    ap.add_argument("--min-edge", type=float, default=0.04)
    ap.add_argument("--cooldown-seconds", type=int, default=600)
    ap.add_argument("--tolerance-seconds", type=int, default=600)
    ap.add_argument(
        "--sweep-min-edges",
        nargs="*",
        default=[],
        help="Optional min-edge values to sweep, e.g. '0.02,0.04,0.06' or '0.02 0.04'.",
    )
    ap.add_argument("--refresh-prices", action="store_true")
    args = ap.parse_args()

    client = PolymarketClient(secrets=load_secrets())
    games = _games(args.feature_cache)
    min_edges = _parse_edges(args.sweep_min_edges) or [args.min_edge]
    summary_rows: list[dict] = []
    all_trades: list[pd.DataFrame] = []

    for game in games.itertuples(index=False):
        market = discover_moneyline(client, game)
        if not market or not market["home_token"] or not market["away_token"]:
            summary_rows.append({"game_id": game.game_id, "status": "no_market"})
            continue

        for outcome, token_id in [("home", market["home_token"]), ("away", market["away_token"])]:
            prices = _fetch_prices(
                client,
                token_id,
                game.date,
                _price_cache_name(game.game_id, outcome),
                args.refresh_prices,
            )
            if prices.empty:
                summary_rows.append(
                    {"game_id": game.game_id, "outcome": outcome, "status": "no_prices"}
                )
                continue
            states, token_won = build_model_states(
                Path(args.kaggle_root),
                game.game_id,
                args.model_path,
                args.feature_cache,
                outcome,
            )
            aligned = align_model_prices(
                states,
                prices,
                tolerance_seconds=args.tolerance_seconds,
            )
            for min_edge in min_edges:
                trades = _run_aligned_backtest(
                    aligned=aligned,
                    token_won=token_won,
                    min_edge=min_edge,
                    cooldown_seconds=args.cooldown_seconds,
                )
                if not trades.empty:
                    trades = trades.copy()
                    trades["game_id"] = game.game_id
                    trades["outcome"] = outcome
                    trades["token_won"] = token_won
                    trades["min_edge"] = min_edge
                    all_trades.append(trades)
                summary_rows.append(
                    {
                        "game_id": game.game_id,
                        "date": game.date,
                        "market_slug": market["market_slug"],
                        "outcome": outcome,
                        "token_won": token_won,
                        "min_edge": min_edge,
                        "status": "ok",
                        "prices": len(prices),
                        "aligned": len(aligned),
                        "trades": len(trades),
                        "stake": float(trades["stake"].sum()) if not trades.empty else 0.0,
                        "pnl": float(trades["pnl"].sum()) if not trades.empty else 0.0,
                    }
                )

    summary = pd.DataFrame(summary_rows)
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    print(summary.to_string(index=False))
    if trades.empty:
        print("\nNo trades fired.")
    else:
        print("\nAggregate by min_edge")
        by_edge = (
            trades.groupby("min_edge", as_index=False)
            .agg(trades=("pnl", "size"), stake=("stake", "sum"), pnl=("pnl", "sum"))
        )
        by_edge["roi"] = by_edge["pnl"] / by_edge["stake"]
        print(by_edge.to_string(index=False, formatters={"roi": "{:+.2%}".format}))

        selected = trades[trades["min_edge"] == args.min_edge]
        if not selected.empty:
            stake = selected["stake"].sum()
            pnl = selected["pnl"].sum()
            print(
                f"\nSelected min_edge={args.min_edge:.3f}: trades={len(selected)} "
                f"stake={stake:.2f} pnl={pnl:+.2f} roi={pnl / stake:+.2%}"
            )
            print(
                selected.groupby(["game_id", "outcome"], as_index=False)
                .agg(trades=("pnl", "size"), stake=("stake", "sum"), pnl=("pnl", "sum"))
                .to_string(index=False)
            )
    if not summary.empty:
        save_parquet(summary, "market_backtest_2024_finals_summary")
    if not trades.empty:
        save_parquet(trades, "market_backtest_2024_finals_trades")


if __name__ == "__main__":
    main()
