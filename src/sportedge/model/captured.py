"""Train/evaluate workflow helpers for live-captured NBA training rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sportedge.model.train import clean_training_rows, train


@dataclass(frozen=True)
class CapturedTrainingSummary:
    path: str
    rows: int
    games: int


def inspect_captured_training(data_path: str) -> CapturedTrainingSummary:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"No captured training cache found at {data_path}")
    df = clean_training_rows(pd.read_parquet(path))
    if "game_id" not in df.columns:
        raise ValueError("captured training cache must include game_id")
    return CapturedTrainingSummary(
        path=str(path),
        rows=int(len(df)),
        games=int(df["game_id"].nunique()),
    )


def train_evaluate_captured(
    data_path: str = "data/cache/training.parquet",
    out_path: str = "models/winprob.joblib",
    min_games: int = 3,
    test_frac: float = 0.2,
    cal_frac: float = 0.2,
) -> str:
    """Train with holdout evaluation from captured rows when enough games exist."""
    summary = inspect_captured_training(data_path)
    if summary.games < min_games:
        raise ValueError(
            f"Need at least {min_games} captured games for train/cal/holdout; "
            f"found {summary.games} games / {summary.rows} rows in {data_path}"
        )
    return train(
        data_path=data_path,
        out_path=out_path,
        test_frac=test_frac,
        cal_frac=cal_frac,
    )
