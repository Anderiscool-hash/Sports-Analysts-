"""Import NocturneBear NBA team totals into a local parquet cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportedge.data.nocturne_nba import load_team_totals  # noqa: E402
from sportedge.data.storage import save_parquet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Import NocturneBear NBA team totals")
    ap.add_argument("--cache", default="nocturne_team_totals_2010_2024")
    ap.add_argument("--regular-only", action="store_true")
    args = ap.parse_args()

    df = load_team_totals(include_playoffs=not args.regular_only)
    path = save_parquet(df, args.cache)
    print(f"saved {len(df):,} team-game rows from {df['game_id'].nunique()} games -> {path}")


if __name__ == "__main__":
    main()
