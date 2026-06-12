"""Kaggle NBA dataset importer.

Dataset: eoinamoore/historical-nba-data-and-player-box-scores

The current archive includes PlayByPlay.parquet plus Games.csv, which is enough
to build the same state-label rows as the live NBA/ESPN scrapers.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.types as pat

from sportedge.data.nba_scraper import HOME_PRIOR, TRAINING_COLUMNS
from sportedge.types import regulation_seconds_remaining

KAGGLE_DATASET = "eoinamoore/historical-nba-data-and-player-box-scores"
PBP_COLUMNS = ["gameId", "clock", "period", "scoreHome", "scoreAway"]
GAME_COLUMNS = ["gameId", "gameDateTimeEst", "hometeamId", "winner", "gameType"]


def download_dataset() -> Path:
    import kagglehub

    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def _clock_to_seconds(clock: str | None) -> float:
    if not clock:
        return 0.0
    match = re.match(r"PT(?:(\d+)M)?(?:([\d.]+)S)?", str(clock))
    if not match:
        return 0.0
    minutes = int(match.group(1) or 0)
    seconds = float(match.group(2) or 0.0)
    return minutes * 60 + seconds


def normalize_game_id(game_id) -> str:
    return str(int(game_id)).zfill(10)


def _load_game_labels(
    root: Path,
    start_date: date,
    end_date: date,
    game_types: tuple[str, ...],
) -> pd.DataFrame:
    games = pd.read_csv(root / "Games.csv", usecols=GAME_COLUMNS)
    games["gameDateTimeEst"] = pd.to_datetime(games["gameDateTimeEst"], errors="coerce")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    games = games[
        (games["gameDateTimeEst"] >= start)
        & (games["gameDateTimeEst"] < end)
        & (games["gameType"].isin(game_types))
    ].copy()
    games["gameId"] = games["gameId"].astype(str)
    games["home_win"] = (games["winner"] == games["hometeamId"]).astype(int)
    return games[["gameId", "home_win"]]


def build_training_set_by_dates(
    root: str | Path,
    start_date: date,
    end_date: date,
    game_types: tuple[str, ...] = ("Regular Season", "Playoffs"),
) -> pd.DataFrame:
    root = Path(root)
    labels = _load_game_labels(root, start_date, end_date, game_types)
    if labels.empty:
        return pd.DataFrame(columns=TRAINING_COLUMNS)

    dataset = ds.dataset(root / "PlayByPlay.parquet", format="parquet")
    game_id_type = dataset.schema.field("gameId").type
    if pat.is_string(game_id_type) or pat.is_large_string(game_id_type):
        game_ids = [str(game_id) for game_id in labels["gameId"].unique()]
    else:
        game_ids = [int(game_id) for game_id in labels["gameId"].unique()]
    pbp = dataset.to_table(
        columns=PBP_COLUMNS,
        filter=ds.field("gameId").isin(game_ids),
    )
    rows = pbp.to_pandas()
    if rows.empty:
        return pd.DataFrame(columns=TRAINING_COLUMNS)

    rows["gameId"] = rows["gameId"].astype(str)
    rows = rows.merge(labels, on="gameId", how="inner")
    rows = rows.dropna(subset=["period", "scoreHome", "scoreAway"])
    rows["period"] = rows["period"].astype(int)
    rows["home_score"] = rows["scoreHome"].fillna(0).astype(int)
    rows["away_score"] = rows["scoreAway"].fillna(0).astype(int)
    rows["seconds_remaining"] = [
        regulation_seconds_remaining(period, _clock_to_seconds(clock))
        for period, clock in zip(rows["period"], rows["clock"], strict=False)
    ]
    rows["game_id"] = rows["gameId"].map(normalize_game_id)
    rows["pre_game_home_prob"] = HOME_PRIOR

    return rows[
        [
            "game_id",
            "home_score",
            "away_score",
            "period",
            "seconds_remaining",
            "pre_game_home_prob",
            "home_win",
        ]
    ].reset_index(drop=True)
