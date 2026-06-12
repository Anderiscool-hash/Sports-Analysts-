"""Tests for the soccer 3-way model-quality backtest (RPS/log-loss/Brier + runner)."""

from __future__ import annotations

import math

from sportedge.model.soccer_backtest import (
    Match,
    backtest,
    brier,
    log_loss,
    outcome_from_score,
    reconstruct_states,
    rps,
)
from sportedge.model.soccer_winprob import SoccerWinProbModel
from sportedge.types import WinProb3


def test_rps_perfect_forecast_is_zero():
    assert rps(WinProb3(1.0, 0.0, 0.0), "home") == 0.0
    assert rps(WinProb3(0.0, 0.0, 1.0), "away") == 0.0


def test_rps_penalises_distance_on_the_ordinal_scale():
    # Predicting away when home occurs (2 steps off) is worse than predicting draw.
    far = rps(WinProb3(0.0, 0.0, 1.0), "home")   # confidently away, home happened
    near = rps(WinProb3(0.0, 1.0, 0.0), "home")  # confidently draw, home happened
    assert far > near > 0.0


def test_rps_uniform_forecast_value():
    # Uniform (1/3,1/3,1/3) with home outcome: cum diffs (2/3, 1/3) -> (4/9+1/9)/2 = 5/18.
    assert math.isclose(rps(WinProb3(1 / 3, 1 / 3, 1 / 3), "home"), 5 / 18, rel_tol=1e-9)


def test_log_loss_and_brier_perfect():
    assert log_loss(WinProb3(1.0, 0.0, 0.0), "home") < 1e-9
    assert brier(WinProb3(1.0, 0.0, 0.0), "home") == 0.0


def test_outcome_from_score():
    assert outcome_from_score(2, 1) == "home"
    assert outcome_from_score(0, 0) == "draw"
    assert outcome_from_score(1, 3) == "away"


def test_reconstruct_states_tracks_running_score():
    states = reconstruct_states(
        "H", "A", goals=[(20, "home"), (70, "away")], lambda_home=1.4, lambda_away=1.1,
        sample_minutes=[0, 30, 90],
    )
    assert (states[0].home_goals, states[0].away_goals) == (0, 0)   # minute 0
    assert (states[1].home_goals, states[1].away_goals) == (1, 0)   # after 20'
    assert (states[2].home_goals, states[2].away_goals) == (1, 1)   # after 70'


def test_match_from_dict_infers_outcome():
    m = Match.from_dict(
        {"home_team": "H", "away_team": "A", "goals": [[10, "home"], [80, "home"]]}
    )
    assert m.outcome == "home"


def _matches() -> list[Match]:
    # Three matches whose goal flow matches the eventual result, so a sane model
    # should score well below the uniform-forecast RPS of 5/18 ~= 0.278.
    return [
        Match("H", "A", "home", [(15, "home"), (60, "home")]),
        Match("H", "A", "away", [(20, "away"), (75, "away")]),
        Match("H", "A", "draw", [(30, "home"), (50, "away")]),
    ]


def test_backtest_aggregates_and_beats_uniform():
    model = SoccerWinProbModel()  # Poisson fallback
    result = backtest(model, _matches())
    assert result.n_forecasts > 0
    assert 0.0 <= result.mean_rps < 5 / 18  # better than always-uniform
    assert result.mean_log_loss > 0.0
    assert result.rps_by_bucket  # bucketed by minute
    # later buckets should be sharper (lower RPS) than the kickoff bucket
    assert result.rps_by_bucket["75-90"] < result.rps_by_bucket["00-15"]


def test_backtest_reliability_has_entries_per_outcome():
    result = backtest(SoccerWinProbModel(), _matches())
    assert set(result.reliability.keys()) == {"home", "draw", "away"}
    # each reliability cell is (mean_pred, empirical_freq, count) with count > 0
    for cells in result.reliability.values():
        for mean_pred, emp_freq, count in cells:
            assert 0.0 <= mean_pred <= 1.0 and 0.0 <= emp_freq <= 1.0 and count > 0


def test_empty_backtest_is_safe():
    result = backtest(SoccerWinProbModel(), [])
    assert result.n_forecasts == 0 and result.mean_rps == 0.0
