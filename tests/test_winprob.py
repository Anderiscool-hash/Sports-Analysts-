from sportedge.model.live_winprob import WinProbModel, logistic_winprob
from sportedge.types import GameState, regulation_seconds_remaining


def _state(home, away, period, clock_secs, prior=0.5):
    return GameState(
        home_team="H",
        away_team="A",
        home_score=home,
        away_score=away,
        period=period,
        seconds_remaining=regulation_seconds_remaining(period, clock_secs),
        pre_game_home_prob=prior,
    )


def test_start_of_game_returns_prior():
    p = logistic_winprob(_state(0, 0, 1, 720, prior=0.65))
    assert abs(p - 0.65) < 0.02


def test_bigger_lead_is_more_confident():
    base = logistic_winprob(_state(50, 45, 4, 300))
    more = logistic_winprob(_state(50, 40, 4, 300))
    assert more > base


def test_late_big_lead_near_certain():
    p = logistic_winprob(_state(100, 90, 4, 10))  # up 10 with 10s left
    assert p > 0.98


def test_trailing_team_below_half():
    p = logistic_winprob(_state(40, 50, 4, 120))
    assert p < 0.5


def test_untrained_model_uses_fallback():
    m = WinProbModel.load("models/does_not_exist.joblib")
    assert not m.is_trained
    assert 0.0 <= m.predict(_state(0, 0, 1, 720)) <= 1.0
