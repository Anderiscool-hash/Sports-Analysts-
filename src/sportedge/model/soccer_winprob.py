"""Live in-game 3-way (1X2) soccer win-probability model.

The model treats each team's remaining goals as an independent Poisson process. Given
the current scoreline and the minutes left, the remaining goals for each side are
``Poisson(lambda * remaining/90)``. The final-result distribution — P(home win),
P(draw), P(away win) on the 90-minute regulation result — is obtained by summing the
joint pmf over the difference of those two Poissons added to the current margin.

This is closed-form and needs no in-game training data, so the whole pipeline runs
immediately. The full-match ``lambda`` priors come from pre-match calibration
(`soccer_calibrate.py`); red cards and a small trailing-team push adjust the effective
in-game rates. A fitted estimator can be swapped in later behind the same interface.
"""

from __future__ import annotations

import math

from sportedge.types import REGULATION_MINUTES, SoccerGameState, WinProb3

# Max remaining goals per team to sum over. WC scoring rates make P(>10 more goals
# from one side in a partial match) astronomically small, so this is exact in practice.
_MAX_GOALS = 12

# Each red card multiplies the carded team's remaining scoring rate (down) and nudges
# the opponent's up. Deliberately conservative for v1.
_RED_CARD_SELF = 0.75
_RED_CARD_OPP = 1.10

# Trailing teams attack more. Effective rate gets a small bump per goal of deficit.
_TRAIL_PUSH_PER_GOAL = 0.06
_TRAIL_PUSH_MAX = 0.30


def _poisson_pmf(k: int, mu: float) -> float:
    if mu <= 0.0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1))


def _trail_multiplier(own_goals: int, opp_goals: int) -> float:
    deficit = max(0, opp_goals - own_goals)
    return 1.0 + min(_TRAIL_PUSH_MAX, _TRAIL_PUSH_PER_GOAL * deficit)


def effective_lambdas(state: SoccerGameState) -> tuple[float, float]:
    """Remaining-match expected goals for (home, away), after scaling the full-match
    priors by the fraction of the match left and applying red-card / trailing tweaks."""
    frac = state.minutes_remaining / REGULATION_MINUTES
    mu_home = state.lambda_home * frac
    mu_away = state.lambda_away * frac

    # Red cards: own cards suppress, opponent cards lift.
    mu_home *= _RED_CARD_SELF**state.home_red_cards * _RED_CARD_OPP**state.away_red_cards
    mu_away *= _RED_CARD_SELF**state.away_red_cards * _RED_CARD_OPP**state.home_red_cards

    # Trailing push.
    mu_home *= _trail_multiplier(state.home_goals, state.away_goals)
    mu_away *= _trail_multiplier(state.away_goals, state.home_goals)
    return mu_home, mu_away


def win_prob_3(
    goal_diff: int,
    mu_home: float,
    mu_away: float,
    max_goals: int = _MAX_GOALS,
) -> WinProb3:
    """1X2 distribution given the current home-minus-away ``goal_diff`` and the
    remaining-goal Poisson means for each side. Probabilities sum to 1."""
    home_pmf = [_poisson_pmf(k, mu_home) for k in range(max_goals + 1)]
    away_pmf = [_poisson_pmf(k, mu_away) for k in range(max_goals + 1)]

    p_home = p_draw = p_away = 0.0
    for gh, ph in enumerate(home_pmf):
        for ga, pa in enumerate(away_pmf):
            joint = ph * pa
            final_margin = goal_diff + gh - ga
            if final_margin > 0:
                p_home += joint
            elif final_margin == 0:
                p_draw += joint
            else:
                p_away += joint

    total = p_home + p_draw + p_away
    if total <= 0.0:  # numerical guard; should not happen
        return WinProb3(1 / 3, 1 / 3, 1 / 3)
    return WinProb3(p_home / total, p_draw / total, p_away / total)


def poisson_winprob(state: SoccerGameState) -> WinProb3:
    """Closed-form 1X2 fair value for a live soccer state. No training required."""
    mu_home, mu_away = effective_lambdas(state)
    return win_prob_3(state.goal_diff, mu_home, mu_away)


class SoccerWinProbModel:
    """Wraps an optional fitted estimator; falls back to the Poisson model if absent.

    Mirrors the NBA ``WinProbModel`` interface so the loop treats both sports the same.
    """

    def __init__(self, estimator=None):
        self._estimator = estimator

    @classmethod
    def load(cls, path: str) -> SoccerWinProbModel:
        from pathlib import Path

        if not path:
            return cls(None)
        p = Path(path)
        if not p.is_file():
            return cls(None)
        import joblib  # local import: optional until a model exists

        return cls(joblib.load(p))

    @property
    def is_trained(self) -> bool:
        return self._estimator is not None

    def predict(self, state: SoccerGameState) -> WinProb3:
        """P(home) / P(draw) / P(away) for the 90-minute regulation result."""
        if self._estimator is None:
            return poisson_winprob(state)
        try:
            from sportedge.model.soccer_features import (
                features_to_vector,
                state_to_features,
            )
        except ImportError:
            # A fitted soccer feature pipeline is optional. Never let a stale or
            # incorrectly configured artifact break the closed-form live model.
            return poisson_winprob(state)

        x = features_to_vector(state_to_features(state)).reshape(1, -1)
        probs = self._estimator.predict_proba(x)[0]
        # Estimator classes are ordered [away, draw, home] = [0, 1, 2] by convention.
        return WinProb3(home=float(probs[2]), draw=float(probs[1]), away=float(probs[0]))
