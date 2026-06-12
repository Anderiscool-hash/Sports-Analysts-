import pandas as pd

from sportedge.model.train import split_by_game


def _rows(game_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [game_id, game_id],
            "home_score": [10, 20],
            "away_score": [8, 15],
            "period": [1, 2],
            "seconds_remaining": [1800.0, 1200.0],
            "pre_game_home_prob": [0.6, 0.6],
            "home_win": [1, 1],
        }
    )


def test_train_split_keeps_games_disjoint():
    df = pd.concat([_rows(f"g{i}") for i in range(20)], ignore_index=True)
    train, cal, holdout = split_by_game(df, test_frac=0.2, cal_frac=0.2, seed=1)

    train_ids = set(train["game_id"])
    cal_ids = set(cal["game_id"])
    holdout_ids = set(holdout["game_id"])

    assert train_ids
    assert cal_ids
    assert holdout_ids
    assert train_ids.isdisjoint(cal_ids)
    assert train_ids.isdisjoint(holdout_ids)
    assert cal_ids.isdisjoint(holdout_ids)
    assert train["game_id"].nunique() == 12
    assert cal["game_id"].nunique() == 4
    assert holdout["game_id"].nunique() == 4


def test_train_split_forces_holdout_games():
    df = pd.concat([_rows(f"g{i}") for i in range(20)], ignore_index=True)
    train, cal, holdout = split_by_game(
        df,
        holdout_game_ids=["g0", "g1", "missing"],
        test_frac=0.2,
        cal_frac=0.2,
        seed=1,
    )

    assert {"g0", "g1"}.issubset(set(holdout["game_id"]))
    assert "g0" not in set(train["game_id"])
    assert "g1" not in set(cal["game_id"])
