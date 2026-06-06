"""Pull historical playoff play-by-play into a cached training set.

    python scripts/fetch_historical.py --seasons 2023-24 2022-23 2021-22

Use Playoffs (default) or Regular Season. Output: data/cache/training.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running as a plain script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportedge.data.nba_scraper import build_training_set  # noqa: E402
from sportedge.data.storage import save_parquet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch NBA training data")
    ap.add_argument("--seasons", nargs="+", default=["2023-24", "2022-23", "2021-22"])
    ap.add_argument("--season-type", default="Playoffs", choices=["Playoffs", "Regular Season"])
    args = ap.parse_args()

    print(f"Fetching {args.season_type} for {args.seasons} … (nba_api can be slow)")
    df = build_training_set(args.seasons, args.season_type)
    if df.empty:
        print("No rows fetched. Check connectivity / nba_api availability.")
        return
    path = save_parquet(df, "training")
    print(f"saved {len(df):,} rows from {df['game_id'].nunique()} games -> {path}")


if __name__ == "__main__":
    main()
