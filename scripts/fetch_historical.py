"""Pull historical play-by-play into a cached training set.

    python scripts/fetch_historical.py --seasons 2023-24 2022-23 2021-22
    python scripts/fetch_historical.py --source espn --start-date 2024-06-06 --end-date 2024-06-06

Use nba_api by season or ESPN by date range. Output defaults to
data/cache/training.parquet.
"""

from __future__ import annotations

import argparse
from datetime import date
import sys
from pathlib import Path

# allow running as a plain script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportedge.data.espn_scraper import build_training_set_by_dates  # noqa: E402
from sportedge.data.nba_scraper import build_training_set  # noqa: E402
from sportedge.data.storage import save_parquet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch NBA training data")
    ap.add_argument("--source", default="nba_api", choices=["nba_api", "espn"])
    ap.add_argument("--seasons", nargs="+", default=["2023-24", "2022-23", "2021-22"])
    ap.add_argument("--season-type", default="Playoffs", choices=["Playoffs", "Regular Season"])
    ap.add_argument("--start-date", default=None, help="YYYY-MM-DD; required for --source espn")
    ap.add_argument("--end-date", default=None, help="YYYY-MM-DD; required for --source espn")
    ap.add_argument("--cache", default="training")
    args = ap.parse_args()

    if args.source == "espn":
        if not args.start_date or not args.end_date:
            ap.error("--source espn requires --start-date and --end-date")
        start = date.fromisoformat(args.start_date)
        end = date.fromisoformat(args.end_date)
        print(f"Fetching ESPN NBA games from {start} through {end} ...")
        df = build_training_set_by_dates(start, end)
    else:
        print(f"Fetching {args.season_type} for {args.seasons} ... (nba_api can be slow)")
        df = build_training_set(args.seasons, args.season_type)

    if df.empty:
        print("No rows fetched. Check connectivity / source availability.")
        return
    path = save_parquet(df, args.cache)
    print(f"saved {len(df):,} rows from {df['game_id'].nunique()} games -> {path}")


if __name__ == "__main__":
    main()
