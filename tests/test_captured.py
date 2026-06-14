import pandas as pd
import pytest

from sportedge.model.captured import inspect_captured_training, train_evaluate_captured


def test_inspect_captured_training_counts_rows_and_games(tmp_path):
    path = tmp_path / "training.parquet"
    pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2"],
            "home_score": [1, 2, 3],
        }
    ).to_parquet(path, index=False)

    summary = inspect_captured_training(str(path))

    assert summary.rows == 3
    assert summary.games == 2


def test_inspect_captured_training_reports_usable_clean_rows(tmp_path):
    path = tmp_path / "training.parquet"
    pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "home_score": [0, 10],
            "away_score": [0, 8],
            "period": [3, 1],
            "seconds_remaining": [600.0, 2600.0],
            "pre_game_home_prob": [0.5, 0.5],
            "home_win": [1, 1],
        }
    ).to_parquet(path, index=False)

    summary = inspect_captured_training(str(path))

    assert summary.rows == 1
    assert summary.games == 1


def test_train_evaluate_captured_requires_enough_games(tmp_path):
    path = tmp_path / "training.parquet"
    pd.DataFrame(
        {
            "game_id": ["g1"],
            "home_score": [1],
            "away_score": [0],
            "period": [1],
            "seconds_remaining": [100.0],
            "pre_game_home_prob": [0.5],
            "home_win": [1],
        }
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="Need at least 3 captured games"):
        train_evaluate_captured(str(path), str(tmp_path / "model.joblib"))
