"""Align one NBA game's model states to Polymarket prices and simulate P&L.

Example:
    python scripts/run_market_backtest.py \
      --game-id 0042300405 \
      --token-outcome home \
      --prices-cache polymarket_2024_finals_g5_celtics_10m
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.types as pat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportedge.data.kaggle_nba import _clock_to_seconds, normalize_game_id  # noqa: E402
from sportedge.data.storage import load_parquet, save_parquet  # noqa: E402
from sportedge.market.pnl import align_model_prices, simulate_buy_and_hold, trades_frame  # noqa: E402
from sportedge.model.live_winprob import WinProbModel  # noqa: E402
from sportedge.types import GameState, regulation_seconds_remaining  # noqa: E402

KAGGLE_ROOT = Path(
    r"C:\Users\ayala\.cache\kagglehub\datasets\eoinamoore"
    r"\historical-nba-data-and-player-box-scores\versions\505"
)
PBP_COLUMNS = [
    "gameId",
    "clock",
    "period",
    "scoreHome",
    "scoreAway",
    "timeActual",
    "actionNumber",
]


def _raw_game_id(game_id: str) -> str:
    return str(int(game_id))


def _load_game_pbp(root: Path, game_id: str) -> pd.DataFrame:
    dataset = ds.dataset(root / "PlayByPlay.parquet", format="parquet")
    field_type = dataset.schema.field("gameId").type
    raw = _raw_game_id(game_id)
    filter_value = raw if pat.is_string(field_type) or pat.is_large_string(field_type) else int(raw)
    table = dataset.to_table(columns=PBP_COLUMNS, filter=ds.field("gameId") == filter_value)
    rows = table.to_pandas()
    if rows.empty:
        raise SystemExit(f"No Kaggle play-by-play rows found for {game_id}")
    return rows


def _game_features(game_id: str, feature_cache: str) -> tuple[int, float, float]:
    rows = load_parquet(feature_cache)
    if rows is None or rows.empty:
        raise SystemExit(f"No rows in feature cache {feature_cache}")
    game_rows = rows[rows["game_id"] == game_id]
    if game_rows.empty:
        raise SystemExit(f"Game {game_id} not found in feature cache {feature_cache}")
    row = game_rows.iloc[0]
    return (
        int(row["home_win"]),
        float(row.get("home_recent_net_rating", 0.0)),
        float(row.get("away_recent_net_rating", 0.0)),
    )


def build_model_states(
    root: Path,
    game_id: str,
    model_path: str,
    feature_cache: str,
    token_outcome: str,
) -> tuple[pd.DataFrame, bool]:
    home_win, home_form, away_form = _game_features(game_id, feature_cache)
    token_won = bool(home_win) if token_outcome == "home" else not bool(home_win)
    model = WinProbModel.load(model_path)

    rows = _load_game_pbp(root, game_id)
    rows = rows.dropna(subset=["timeActual", "period", "scoreHome", "scoreAway"]).copy()
    parsed_time = pd.to_datetime(rows["timeActual"], utc=True, errors="coerce")
    rows["timestamp"] = parsed_time.map(lambda value: int(value.timestamp()) if pd.notna(value) else pd.NA)
    rows = rows[rows["timestamp"].notna()]
    rows["actionNumber"] = rows["actionNumber"].fillna(0)
    rows = rows.sort_values(["timestamp", "actionNumber"]).drop_duplicates("timestamp", keep="last")

    out_rows: list[dict] = []
    for row in rows.itertuples(index=False):
        period = int(row.period)
        state = GameState(
            home_team="HOME",
            away_team="AWAY",
            home_score=int(row.scoreHome or 0),
            away_score=int(row.scoreAway or 0),
            period=period,
            seconds_remaining=regulation_seconds_remaining(period, _clock_to_seconds(row.clock)),
            pre_game_home_prob=0.60,
            home_recent_net_rating=home_form,
            away_recent_net_rating=away_form,
        )
        home_p = model.predict(state)
        model_p = home_p if token_outcome == "home" else 1.0 - home_p
        out_rows.append(
            {
                "game_id": game_id,
                "timestamp": int(row.timestamp),
                "home_score": state.home_score,
                "away_score": state.away_score,
                "period": state.period,
                "seconds_remaining": state.seconds_remaining,
                "model_p": model_p,
                "token_outcome": token_outcome,
            }
        )
    return pd.DataFrame(out_rows), token_won


def run_market_backtest(
    *,
    game_id: str,
    token_outcome: str,
    prices: pd.DataFrame,
    feature_cache: str = "training_kaggle_2021_24_with_team_form",
    kaggle_root: Path = KAGGLE_ROOT,
    model_path: str = "models/winprob.joblib",
    min_edge: float = 0.04,
    bankroll: float = 100.0,
    kelly_fraction: float = 0.25,
    max_stake: float = 5.0,
    cooldown_seconds: int = 600,
    tolerance_seconds: int = 600,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    states, token_won = build_model_states(
        kaggle_root,
        game_id,
        model_path,
        feature_cache,
        token_outcome,
    )
    aligned = align_model_prices(states, prices, tolerance_seconds=tolerance_seconds)
    trades = simulate_buy_and_hold(
        aligned,
        token_won=token_won,
        min_edge=min_edge,
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        max_stake=max_stake,
        cooldown_seconds=cooldown_seconds,
    )
    return aligned, trades_frame(trades), token_won


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one-game market P&L backtest")
    ap.add_argument("--game-id", required=True, help="10-digit NBA game id")
    ap.add_argument("--token-outcome", choices=["home", "away"], required=True)
    ap.add_argument("--prices-cache", required=True)
    ap.add_argument("--feature-cache", default="training_kaggle_2021_24_with_team_form")
    ap.add_argument("--kaggle-root", default=str(KAGGLE_ROOT))
    ap.add_argument("--model-path", default="models/winprob.joblib")
    ap.add_argument("--min-edge", type=float, default=0.04)
    ap.add_argument("--bankroll", type=float, default=100.0)
    ap.add_argument("--kelly-fraction", type=float, default=0.25)
    ap.add_argument("--max-stake", type=float, default=5.0)
    ap.add_argument("--cooldown-seconds", type=int, default=600)
    ap.add_argument("--tolerance-seconds", type=int, default=600)
    ap.add_argument("--out-cache", default=None)
    args = ap.parse_args()

    game_id = normalize_game_id(args.game_id)
    prices = load_parquet(args.prices_cache)
    if prices is None or prices.empty:
        raise SystemExit(f"No rows in prices cache {args.prices_cache}")

    aligned, trade_df, token_won = run_market_backtest(
        game_id=game_id,
        token_outcome=args.token_outcome,
        prices=prices,
        feature_cache=args.feature_cache,
        kaggle_root=Path(args.kaggle_root),
        model_path=args.model_path,
        min_edge=args.min_edge,
        bankroll=args.bankroll,
        kelly_fraction=args.kelly_fraction,
        max_stake=args.max_stake,
        cooldown_seconds=args.cooldown_seconds,
        tolerance_seconds=args.tolerance_seconds,
    )

    print(f"aligned: {len(aligned):,}")
    print(f"token_outcome={args.token_outcome} token_won={token_won}")
    if trade_df.empty:
        print("trades: 0")
    else:
        print(
            f"trades: {len(trade_df)}  stake={trade_df['stake'].sum():.2f}  "
            f"pnl={trade_df['pnl'].sum():+.2f}  roi={trade_df['pnl'].sum() / trade_df['stake'].sum():+.2%}"
        )
        print(trade_df.head(20).to_string(index=False))

    if args.out_cache:
        out = aligned.copy()
        out["token_won"] = token_won
        path = save_parquet(out, args.out_cache)
        print(f"wrote aligned states -> {path}")


if __name__ == "__main__":
    main()
