"""Tests for the Poisson 1X2 soccer win-probability model."""

from __future__ import annotations

from sportedge.model.soccer_winprob import (
    effective_lambdas,
    poisson_winprob,
    win_prob_3,
)
from sportedge.types import SoccerGameState


def _sums_to_one(p) -> bool:
    return abs((p.home + p.draw + p.away) - 1.0) < 1e-9


def test_probabilities_sum_to_one():
    p = win_prob_3(goal_diff=0, mu_home=1.4, mu_away=1.1)
    assert _sums_to_one(p)


def test_kickoff_level_match_favours_home_slightly():
    # 0-0 at kickoff with a typical home edge: home > away, draw is meaningful.
    state = SoccerGameState("H", "A", 0, 0, minute=0, lambda_home=1.45, lambda_away=1.15)
    p = poisson_winprob(state)
    assert _sums_to_one(p)
    assert p.home > p.away
    assert 0.2 < p.draw < 0.4


def test_no_time_left_level_is_a_certain_draw():
    # Full time, scores level -> draw is essentially certain.
    state = SoccerGameState("H", "A", 1, 1, minute=90)
    p = poisson_winprob(state)
    assert p.draw > 0.999
    assert p.home < 1e-3 and p.away < 1e-3


def test_no_time_left_lead_is_a_certain_win():
    # Full time, home up one -> home win is essentially certain.
    state = SoccerGameState("H", "A", 2, 1, minute=90)
    p = poisson_winprob(state)
    assert p.home > 0.999


def test_late_one_goal_lead_is_strong_but_not_certain():
    # 1-0 home with 5 minutes left: heavily favoured, draw still possible, away tiny.
    state = SoccerGameState("H", "A", 1, 0, minute=85)
    p = poisson_winprob(state)
    assert p.home > 0.85
    assert p.away < 0.05
    assert p.draw > 0.0


def test_zero_minutes_remaining_scales_lambdas_to_zero():
    state = SoccerGameState("H", "A", 0, 0, minute=90)
    mu_home, mu_away = effective_lambdas(state)
    assert mu_home == 0.0 and mu_away == 0.0


def test_stoppage_time_reads_as_no_time_remaining():
    # minute > 90 must clamp to 0 remaining, not go negative.
    state = SoccerGameState("H", "A", 0, 0, minute=94)
    assert state.minutes_remaining == 0.0
    p = poisson_winprob(state)
    assert p.draw > 0.999


def test_red_card_against_home_lowers_home_scoring_rate():
    base = SoccerGameState("H", "A", 0, 0, minute=45)
    carded = SoccerGameState("H", "A", 0, 0, minute=45, home_red_cards=1)
    mu_home_base, _ = effective_lambdas(base)
    mu_home_carded, _ = effective_lambdas(carded)
    assert mu_home_carded < mu_home_base


def test_red_card_against_away_helps_home_winprob():
    base = SoccerGameState("H", "A", 0, 0, minute=45)
    away_carded = SoccerGameState("H", "A", 0, 0, minute=45, away_red_cards=1)
    assert poisson_winprob(away_carded).home > poisson_winprob(base).home


def test_trailing_team_gets_a_scoring_push():
    # Home trailing 0-1 should attack more than the symmetric level-game rate.
    trailing = SoccerGameState("H", "A", 0, 1, minute=45, lambda_home=1.3, lambda_away=1.3)
    level = SoccerGameState("H", "A", 0, 0, minute=45, lambda_home=1.3, lambda_away=1.3)
    mu_home_trailing, _ = effective_lambdas(trailing)
    mu_home_level, _ = effective_lambdas(level)
    assert mu_home_trailing > mu_home_level


def test_bigger_lead_means_higher_home_winprob():
    one = SoccerGameState("H", "A", 1, 0, minute=60)
    two = SoccerGameState("H", "A", 2, 0, minute=60)
    assert poisson_winprob(two).home > poisson_winprob(one).home
