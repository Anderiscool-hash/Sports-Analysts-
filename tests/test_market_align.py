import pandas as pd

from sportedge.market.align import align_price_rows, parse_raw_price_path


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, state):
        return self.value


def _training_rows():
    return pd.DataFrame(
        {
            "game_id": ["0042300401", "0042300401", "0042300401"],
            "home_score": [10, 20, 30],
            "away_score": [8, 22, 28],
            "period": [1, 2, 4],
            "seconds_remaining": [2800, 1800, 0],
            "pre_game_home_prob": [0.55, 0.55, 0.55],
            "home_win": [1, 1, 1],
        }
    )


def test_parse_raw_price_path():
    parsed = parse_raw_price_path("data/cache/polymarket_0042300401_home_10m.parquet")

    assert parsed is not None
    assert parsed.game_id == "0042300401"
    assert parsed.side == "home"
    assert parsed.window == "10m"
    assert parse_raw_price_path("data/cache/aligned_0042300401_home.parquet") is None


def test_align_price_rows_for_home_side():
    prices = pd.DataFrame(
        {
            "token_id": ["home-token", "home-token", "home-token"],
            "timestamp": [1000, 1100, 1200],
            "price": [0.60, 0.50, 0.55],
        }
    )

    aligned = align_price_rows(_training_rows(), prices, "0042300401", "home", ConstantModel(0.70))

    assert list(aligned["token_outcome"].unique()) == ["home"]
    assert list(aligned["token_id"].unique()) == ["home-token"]
    assert aligned["model_p"].tolist() == [0.70, 0.70, 0.70]
    assert aligned["token_won"].tolist() == [1, 1, 1]
    assert round(float(aligned.iloc[0]["edge"]), 2) == 0.10


def test_align_price_rows_for_away_side_flips_probability_and_result():
    prices = pd.DataFrame(
        {
            "token_id": ["away-token", "away-token", "away-token"],
            "timestamp": [1000, 1100, 1200],
            "price": [0.40, 0.45, 0.50],
        }
    )

    aligned = align_price_rows(_training_rows(), prices, "0042300401", "away", ConstantModel(0.70))

    assert list(aligned["token_outcome"].unique()) == ["away"]
    assert list(aligned["token_id"].unique()) == ["away-token"]
    assert [round(value, 2) for value in aligned["model_p"].tolist()] == [0.30, 0.30, 0.30]
    assert aligned["token_won"].tolist() == [0, 0, 0]
    assert round(float(aligned.iloc[0]["edge"]), 2) == -0.10
