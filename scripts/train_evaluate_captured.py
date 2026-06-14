"""Train and evaluate the NBA model from live-captured dashboard rows.

    python scripts/train_evaluate_captured.py --data data/cache/training.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportedge.model.captured import inspect_captured_training, train_evaluate_captured  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/evaluate from captured live NBA games")
    ap.add_argument("--data", default="data/cache/training.parquet")
    ap.add_argument("--out", default="models/winprob.joblib")
    ap.add_argument("--min-games", type=int, default=3)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--cal-frac", type=float, default=0.2)
    args = ap.parse_args()

    summary = inspect_captured_training(args.data)
    print(f"captured cache: {summary.rows:,} rows from {summary.games:,} games -> {summary.path}")
    train_evaluate_captured(
        data_path=args.data,
        out_path=args.out,
        min_games=args.min_games,
        test_frac=args.test_frac,
        cal_frac=args.cal_frac,
    )


if __name__ == "__main__":
    main()
