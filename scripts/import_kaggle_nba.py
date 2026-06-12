"""Import Kaggle NBA play-by-play into SportEdge training rows.

    python scripts/import_kaggle_nba.py --start-date 2021-10-19 --end-date 2024-06-17
"""

from __future__ import annotations

import argparse
from datetime import date
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportedge.data.kaggle_nba import build_training_set_by_dates, download_dataset  # noqa: E402
from sportedge.data.storage import save_parquet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Import Kaggle NBA play-by-play")
    ap.add_argument("--root", default=None, help="Path containing Games.csv and PlayByPlay.parquet")
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument(
        "--game-types",
        nargs="+",
        default=["Regular Season", "Playoffs"],
        help="Game types to include, e.g. Regular Season Playoffs",
    )
    ap.add_argument("--cache", default="training_kaggle_nba")
    args = ap.parse_args()

    root = Path(args.root) if args.root else download_dataset()
    print(f"Using Kaggle dataset at {root}")
    df = build_training_set_by_dates(
        root,
        date.fromisoformat(args.start_date),
        date.fromisoformat(args.end_date),
        tuple(args.game_types),
    )
    if df.empty:
        print("No rows imported. Check date range and game types.")
        return
    path = save_parquet(df, args.cache)
    print(f"saved {len(df):,} rows from {df['game_id'].nunique()} games -> {path}")


if __name__ == "__main__":
    main()
