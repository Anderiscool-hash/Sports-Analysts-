"""Tests for soccer pre-match calibration (attack/defense/home-adv -> lambda priors)."""

from __future__ import annotations

from sportedge.model.soccer_calibrate import MatchResult, TeamRatings, fit_ratings
from sportedge.model.soccer_winprob import win_prob_3


def _synthetic_results() -> list[MatchResult]:
    # "Strong" thrashes "Weak"; "Mid" sits in between. Repeated to give the fit signal.
    games = []
    for _ in range(40):
        games.append(MatchResult("Strong", "Weak", 3, 0))
        games.append(MatchResult("Weak", "Strong", 0, 3))
        games.append(MatchResult("Strong", "Mid", 2, 1))
        games.append(MatchResult("Mid", "Weak", 2, 0))
        games.append(MatchResult("Mid", "Strong", 1, 2))
        games.append(MatchResult("Weak", "Mid", 0, 2))
    return games


def test_strong_team_gets_higher_attack_than_weak():
    ratings = fit_ratings(_synthetic_results())
    assert ratings.attack["Strong"] > ratings.attack["Mid"] > ratings.attack["Weak"]


def test_strong_team_has_higher_expected_goals():
    ratings = fit_ratings(_synthetic_results())
    lam_strong_home, lam_weak_away = ratings.lambdas("Strong", "Weak")
    lam_weak_home, lam_strong_away = ratings.lambdas("Weak", "Strong")
    assert lam_strong_home > lam_weak_away
    assert lam_strong_away > lam_weak_home


def test_home_advantage_detected_when_present():
    # Home-tilted data: the home side consistently outscores the away side for an
    # otherwise even matchup. The fit should recover a positive home advantage.
    games = [MatchResult("X", "Y", 2, 1) for _ in range(40)]
    games += [MatchResult("Y", "X", 2, 1) for _ in range(40)]
    ratings = fit_ratings(games)
    assert ratings.home_adv > 0.0


def test_symmetric_data_has_near_zero_home_advantage():
    # Neutral-venue style (WC): mirrored results -> home advantage ~ 0.
    ratings = fit_ratings(_synthetic_results())
    assert abs(ratings.home_adv) < 0.05


def test_calibrated_lambdas_feed_the_winprob_model():
    # End-to-end: fitted lambdas -> kickoff 1X2 strongly favours the stronger home side.
    ratings = fit_ratings(_synthetic_results())
    lam_home, lam_away = ratings.lambdas("Strong", "Weak")
    p = win_prob_3(goal_diff=0, mu_home=lam_home, mu_away=lam_away)
    assert p.home > p.away
    assert p.home > 0.5


def test_unknown_team_falls_back_to_league_average():
    ratings = fit_ratings(_synthetic_results())
    lam_home, lam_away = ratings.lambdas("Nowhere United", "Parts Unknown")
    assert lam_home > 0.0 and lam_away > 0.0


def test_ratings_json_round_trip(tmp_path):
    ratings = fit_ratings(_synthetic_results())
    path = tmp_path / "ratings.json"
    ratings.to_json(str(path))
    loaded = TeamRatings.from_json(str(path))
    assert loaded.lambdas("Strong", "Weak") == ratings.lambdas("Strong", "Weak")


def test_empty_results_returns_default_ratings():
    ratings = fit_ratings([])
    lam_home, lam_away = ratings.lambdas("A", "B")
    assert lam_home > 0.0 and lam_away > 0.0
