"""Attach NocturneBear rolling team-form features to a training cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportedge.data.nocturne_nba import enrich_training_rows_with_team_form  # noqa: E402
from sportedge.data.storage import load_parquet, save_parquet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich training rows with Nocturne team form")
    ap.add_argument("--training-cache", default="training_kaggle_2021_24_regular_plus_playoffs")
    ap.add_argument("--team-cache", default="nocturne_team_totals_2010_2024")
    ap.add_argument("--out-cache", default="training_kaggle_2021_24_with_team_form")
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()

    training = load_parquet(args.training_cache)
    team_totals = load_parquet(args.team_cache)
    if training is None or training.empty:
        raise SystemExit(f"No rows in training cache {args.training_cache}")
    if team_totals is None or team_totals.empty:
        raise SystemExit(f"No rows in team cache {args.team_cache}")

    enriched = enrich_training_rows_with_team_form(training, team_totals, window=args.window)
    path = save_parquet(enriched, args.out_cache)
    matched = (enriched["home_recent_net_rating"].ne(0) | enriched["away_recent_net_rating"].ne(0)).sum()
    print(
        f"saved {len(enriched):,} rows from {enriched['game_id'].nunique()} games -> {path}"
    )
    print(f"team-form coverage: {matched:,}/{len(enriched):,} rows")


if __name__ == "__main__":
    main()
