from sportedge.data.nocturne_nba import normalize_game_id
from sportedge.data.nocturne_nba import enrich_training_rows_with_team_form

import pandas as pd


def test_normalize_game_id_zero_pads_regular_season_ids():
    assert normalize_game_id(22201230) == "0022201230"


def test_normalize_game_id_zero_pads_playoff_ids():
    assert normalize_game_id(42100401) == "0042100401"


def test_enrich_training_rows_uses_prior_team_games_only():
    team_totals = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"],
            "game_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-04",
                ]
            ),
            "team_id": [1, 2, 1, 2, 1, 2, 1, 2],
            "is_home": [True, False, False, True, True, False, True, False],
            "plus_minus": [10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 100.0, -100.0],
        }
    )
    training = pd.DataFrame(
        {
            "game_id": ["g4"],
            "home_score": [0],
            "away_score": [0],
            "period": [1],
            "seconds_remaining": [2880.0],
            "pre_game_home_prob": [0.6],
            "home_win": [1],
        }
    )

    rows = enrich_training_rows_with_team_form(training, team_totals, window=3)

    assert rows.iloc[0]["home_recent_net_rating"] == 20.0
    assert rows.iloc[0]["away_recent_net_rating"] == -20.0
    assert rows.iloc[0]["recent_net_rating_diff"] == 40.0
