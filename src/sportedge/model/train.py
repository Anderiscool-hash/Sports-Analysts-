"""Train + calibrate the live win-probability model from cached training rows.

    python -m sportedge.model.train --data data/cache/training.parquet

Calibration matters: the output is compared directly to market prices, so reliable
probabilities beat raw accuracy. Falls back gracefully if xgboost is unavailable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sportedge.model.features import FEATURE_NAMES, features_to_vector, state_to_features
from sportedge.types import GameState


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
        )
        X.append(features_to_vector(state_to_features(st)))
        y.append(int(r.home_win))
    return np.asarray(X), np.asarray(y)


def train(data_path: str, out_path: str) -> str:
    import joblib
    import pandas as pd
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(data_path)
    if df.empty:
        raise SystemExit(f"No training rows in {data_path}. Run fetch_historical first.")
    X, y = rows_to_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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
    model = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    model.fit(X_te, y_te)

    from sklearn.metrics import brier_score_loss, log_loss

    p = model.predict_proba(X_te)[:, 1]
    print(f"features: {FEATURE_NAMES}")
    print(f"holdout log_loss={log_loss(y_te, p):.4f} brier={brier_score_loss(y_te, p):.4f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"saved -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Train live win-prob model")
    ap.add_argument("--data", default="data/cache/training.parquet")
    ap.add_argument("--out", default="models/live_winprob.joblib")
    args = ap.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()
