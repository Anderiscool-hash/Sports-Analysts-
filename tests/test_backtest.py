import pandas as pd

from sportedge.model.backtest import (
    evaluate,
    evaluate_overall_and_subset,
    split_by_game,
)

COLUMNS = [
    "game_id",
    "home_score",
    "away_score",
    "period",
    "seconds_remaining",
    "pre_game_home_prob",
    "home_win",
]


def _rows(game_id: str, home_win: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [game_id, game_id, game_id],
            "home_score": [10, 20, 30],
            "away_score": [8, 15, 25],
            "period": [1, 2, 4],
            "seconds_remaining": [1800.0, 1200.0, 5.0],
            "pre_game_home_prob": [0.6, 0.6, 0.6],
            "home_win": [home_win, home_win, home_win],
        },
        columns=COLUMNS,
    )


def _frame(n_games: int = 10) -> pd.DataFrame:
    return pd.concat(
        [_rows(f"g{i}", i % 2) for i in range(n_games)], ignore_index=True
    )


class _StubModel:
    is_trained = False

    def predict(self, state) -> float:
        return 0.5


def test_split_disjoint_games():
    df = _frame(10)
    train, test = split_by_game(df, test_frac=0.3, seed=1)
    assert set(train["game_id"]).isdisjoint(set(test["game_id"]))


def test_split_finals_always_in_test():
    df = _frame(10)
    train, test = split_by_game(df, finals_game_ids=["g0", "g1"], test_frac=0.3, seed=1)
    assert {"g0", "g1"}.issubset(set(test["game_id"]))
    assert "g0" not in set(train["game_id"])


def test_split_test_frac_honored():
    df = _frame(10)
    _, test = split_by_game(df, test_frac=0.3, seed=1)
    assert test["game_id"].nunique() == 3


def test_evaluate_stub_model_deterministic():
    df = _frame(4)
    report = evaluate(_StubModel(), df, "overall")
    assert report.n_games == 4
    assert report.n_states == 12
    assert report.brier == 0.25  # all predictions 0.5


def test_evaluate_overall_and_subset_finals_key():
    df = _frame(10)
    reports = evaluate_overall_and_subset(_StubModel(), df, finals_game_ids=["g0"])
    assert "overall" in reports
    assert "finals" in reports
    assert reports["finals"].n_games == 1


def test_evaluate_overall_and_subset_no_finals():
    df = _frame(10)
    reports = evaluate_overall_and_subset(_StubModel(), df, finals_game_ids=[])
    assert "overall" in reports
    assert "finals" not in reports
