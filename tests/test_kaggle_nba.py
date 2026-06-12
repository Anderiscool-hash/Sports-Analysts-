from datetime import date

import pandas as pd

from sportedge.data.kaggle_nba import build_training_set_by_dates, normalize_game_id
from sportedge.data.nba_scraper import TRAINING_COLUMNS


def test_normalize_game_id_zero_pads_nba_ids():
    assert normalize_game_id(42100401) == "0042100401"


def test_build_training_set_by_dates(tmp_path):
    pd.DataFrame(
        {
            "gameId": [42100401, 42100402],
            "gameDateTimeEst": ["2022-06-02 21:00:00", "2022-07-10 21:00:00"],
            "hometeamId": [1, 1],
            "winner": [1, 2],
            "gameType": ["Playoffs", "Playoffs"],
        }
    ).to_csv(tmp_path / "Games.csv", index=False)
    pd.DataFrame(
        {
            "gameId": [42100401, 42100401, 42100402],
            "clock": ["PT12M00.00S", "PT11M30.00S", "PT12M00.00S"],
            "period": [1, 1, 1],
            "scoreHome": [0, 2, 0],
            "scoreAway": [0, 0, 0],
        }
    ).to_parquet(tmp_path / "PlayByPlay.parquet", index=False)

    rows = build_training_set_by_dates(
        tmp_path,
        date(2022, 6, 1),
        date(2022, 6, 30),
        ("Playoffs",),
    )

    assert list(rows.columns) == TRAINING_COLUMNS
    assert len(rows) == 2
    assert rows.iloc[0]["game_id"] == "0042100401"
    assert rows.iloc[0]["seconds_remaining"] == 2880.0
    assert rows.iloc[1]["home_score"] == 2
    assert rows.iloc[1]["home_win"] == 1
