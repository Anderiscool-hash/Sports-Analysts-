"""Importer for NocturneBear/NBA-Data-2010-2024 team totals.

This source is box-score/team-total data, not play-by-play. It is useful for
team-form and strength features, but it does not produce in-game state rows by
itself.
"""

from __future__ import annotations

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/NocturneBear/NBA-Data-2010-2024/main"
REGULAR_TOTALS_URL = f"{BASE_URL}/regular_season_totals_2010_2024.csv"
PLAYOFF_TOTALS_URL = f"{BASE_URL}/play_off_totals_2010_2024.csv"

TEAM_TOTAL_COLUMNS = [
    "SEASON_YEAR",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "WL",
    "PTS",
    "PLUS_MINUS",
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
]


def normalize_game_id(game_id) -> str:
    return str(int(game_id)).zfill(10)


def load_team_totals(include_playoffs: bool = True) -> pd.DataFrame:
    frames = [
        pd.read_csv(REGULAR_TOTALS_URL, usecols=TEAM_TOTAL_COLUMNS).assign(game_type="Regular Season")
    ]
    if include_playoffs:
        frames.append(
            pd.read_csv(PLAYOFF_TOTALS_URL, usecols=TEAM_TOTAL_COLUMNS).assign(game_type="Playoffs")
        )
    df = pd.concat(frames, ignore_index=True)
    df["game_id"] = df["GAME_ID"].map(normalize_game_id)
    df["game_date"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["team_id"] = df["TEAM_ID"].astype(int)
    df["is_home"] = df["MATCHUP"].str.contains("vs.", regex=False, na=False)
    df["win"] = df["WL"].eq("W").astype(int)
    return df.rename(
        columns={
            "SEASON_YEAR": "season_year",
            "TEAM_ABBREVIATION": "team_abbrev",
            "TEAM_NAME": "team_name",
            "PTS": "points",
            "PLUS_MINUS": "plus_minus",
            "FG_PCT": "fg_pct",
            "FG3_PCT": "fg3_pct",
            "FT_PCT": "ft_pct",
            "REB": "rebounds",
            "AST": "assists",
            "TOV": "turnovers",
            "STL": "steals",
            "BLK": "blocks",
        }
    )[
        [
            "game_id",
            "game_date",
            "game_type",
            "season_year",
            "team_id",
            "team_abbrev",
            "team_name",
            "is_home",
            "win",
            "points",
            "plus_minus",
            "fg_pct",
            "fg3_pct",
            "ft_pct",
            "rebounds",
            "assists",
            "turnovers",
            "steals",
            "blocks",
        ]
    ]


def rolling_team_form(team_totals: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Return pregame rolling team form keyed by game/team.

    Values are shifted by one game inside each team, so the current game's final
    box score never leaks into its own features.
    """
    df = team_totals.copy()
    df = df.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
    rolling = df.groupby("team_id")["plus_minus"].transform(
        lambda values: values.rolling(window=window, min_periods=3).mean().shift(1)
    )
    df["recent_net_rating"] = rolling
    return df[["game_id", "team_id", "is_home", "recent_net_rating"]]


def enrich_training_rows_with_team_form(
    training: pd.DataFrame,
    team_totals: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """Attach NocturneBear rolling team-strength features to training rows."""
    form = rolling_team_form(team_totals, window=window)
    home = (
        form[form["is_home"]][["game_id", "recent_net_rating"]]
        .rename(columns={"recent_net_rating": "home_recent_net_rating"})
        .drop_duplicates("game_id")
    )
    away = (
        form[~form["is_home"]][["game_id", "recent_net_rating"]]
        .rename(columns={"recent_net_rating": "away_recent_net_rating"})
        .drop_duplicates("game_id")
    )

    enriched = training.merge(home, on="game_id", how="left").merge(away, on="game_id", how="left")
    enriched["home_recent_net_rating"] = enriched["home_recent_net_rating"].fillna(0.0)
    enriched["away_recent_net_rating"] = enriched["away_recent_net_rating"].fillna(0.0)
    enriched["recent_net_rating_diff"] = (
        enriched["home_recent_net_rating"] - enriched["away_recent_net_rating"]
    )
    return enriched
