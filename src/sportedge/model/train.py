"""Train + calibrate the live win-probability model from cached training rows.

    python -m sportedge.model.train --data data/cache/training.parquet

Calibration matters: the output is compared directly to market prices, so reliable
probabilities beat raw accuracy. Falls back gracefully if xgboost is unavailable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sportedge.model.features import FEATURE_NAMES, features_to_vector, state_to_features
from sportedge.types import GameState


def _row_float(row, name: str, default: float = 0.0) -> float:
    value = getattr(row, name, default)
    if pd.isna(value):
        return default
    return float(value)


def rows_to_xy(df) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for r in df.itertuples(index=False):
        st = GameState(
            home_team="H",
            away_team="A",
            home_score=int(r.home_score),
            away_score=int(r.away_score),
            period=int(r.period),
            seconds_remaining=float(r.seconds_remaining),
            pre_game_home_prob=float(r.pre_game_home_prob),
            home_recent_net_rating=_row_float(r, "home_recent_net_rating"),
            away_recent_net_rating=_row_float(r, "away_recent_net_rating"),
        )
        X.append(features_to_vector(state_to_features(st)))
        y.append(int(r.home_win))
    return np.asarray(X), np.asarray(y)


def split_by_game(
    df: pd.DataFrame,
    holdout_game_ids: list[str] | tuple[str, ...] = (),
    test_frac: float = 0.2,
    cal_frac: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows into train/calibration/holdout sets without leaking games."""
    if "game_id" not in df.columns:
        raise ValueError("training data must include game_id")
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be between 0 and 1")
    if not 0.0 < cal_frac < 1.0:
        raise ValueError("cal_frac must be between 0 and 1")
    if test_frac + cal_frac >= 1.0:
        raise ValueError("test_frac + cal_frac must be less than 1")

    game_ids = np.array(pd.unique(df["game_id"]), dtype=object)
    if len(game_ids) < 3:
        raise ValueError("need at least 3 games for train/calibration/holdout splits")

    rng = np.random.default_rng(seed)
    rng.shuffle(game_ids)

    n_games = len(game_ids)
    n_test = max(1, round(test_frac * n_games))
    n_cal = max(1, round(cal_frac * n_games))
    if n_test + n_cal >= n_games:
        n_test = 1
        n_cal = 1

    id_set = set(game_ids.tolist())
    forced_holdout_ids = {game_id for game_id in holdout_game_ids if game_id in id_set}
    remaining_ids = np.array(
        [game_id for game_id in game_ids.tolist() if game_id not in forced_holdout_ids],
        dtype=object,
    )
    n_test_random = max(0, n_test - len(forced_holdout_ids))

    test_ids = forced_holdout_ids | set(remaining_ids[:n_test_random].tolist())
    cal_ids = set(remaining_ids[n_test_random : n_test_random + n_cal].tolist())
    train_ids = set(remaining_ids[n_test_random + n_cal :].tolist())
    if not train_ids or not cal_ids or not test_ids:
        raise ValueError("split produced an empty train/calibration/holdout set")

    return (
        df[df["game_id"].isin(train_ids)].reset_index(drop=True),
        df[df["game_id"].isin(cal_ids)].reset_index(drop=True),
        df[df["game_id"].isin(test_ids)].reset_index(drop=True),
    )


def _calibrated_prefit_model(base, X_cal: np.ndarray, y_cal: np.ndarray):
    from sklearn.calibration import CalibratedClassifierCV

    try:
        from sklearn.frozen import FrozenEstimator

        model = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    except Exception:
        model = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    model.fit(X_cal, y_cal)
    return model


def train(
    data_path: str,
    out_path: str,
    test_frac: float = 0.2,
    cal_frac: float = 0.2,
    holdout_game_ids: list[str] | tuple[str, ...] = (),
) -> str:
    import joblib

    df = pd.read_parquet(data_path)
    if df.empty:
        raise SystemExit(f"No training rows in {data_path}. Run fetch_historical first.")
    train_df, cal_df, holdout_df = split_by_game(
        df,
        holdout_game_ids=holdout_game_ids,
        test_frac=test_frac,
        cal_frac=cal_frac,
    )
    X_tr, y_tr = rows_to_xy(train_df)
    X_cal, y_cal = rows_to_xy(cal_df)
    X_hold, y_hold = rows_to_xy(holdout_df)

    try:
        from xgboost import XGBClassifier

        base = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            eval_metric="logloss",
        )
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier

        base = GradientBoostingClassifier()

    base.fit(X_tr, y_tr)
    model = _calibrated_prefit_model(base, X_cal, y_cal)

    from sklearn.metrics import brier_score_loss, log_loss

    p = model.predict_proba(X_hold)[:, 1]
    print(f"features: {FEATURE_NAMES}")
    print(
        "split: "
        f"train={train_df['game_id'].nunique()} games/{len(train_df)} rows, "
        f"cal={cal_df['game_id'].nunique()} games/{len(cal_df)} rows, "
        f"holdout={holdout_df['game_id'].nunique()} games/{len(holdout_df)} rows"
    )
    print(f"holdout log_loss={log_loss(y_hold, p):.4f} brier={brier_score_loss(y_hold, p):.4f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"saved -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Train live win-prob model")
    ap.add_argument("--data", default="data/cache/training.parquet")
    ap.add_argument("--out", default="models/winprob.joblib")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--cal-frac", type=float, default=0.2)
    ap.add_argument("--holdout-game-ids", nargs="*", default=[])
    args = ap.parse_args()
    train(args.data, args.out, args.test_frac, args.cal_frac, args.holdout_game_ids)


if __name__ == "__main__":
    main()
