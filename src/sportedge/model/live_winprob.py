"""Live in-game win-probability model.

Ships with a transparent logistic *fallback* (closed-form on lead and time) so the
whole pipeline runs before any model is trained. Once `train.py` produces a
calibrated gradient-boosted model, `WinProbModel.load` uses that instead.
"""

from __future__ import annotations

import math
from pathlib import Path

from sportedge.model.features import features_to_vector, state_to_features
from sportedge.types import GameState

# Heuristic coefficient for the fallback. Calibrated by eye to NBA reality
# (e.g. up 10 with 10:00 left ≈ ~88%). Replaced by the trained model when present.
_LEAD_COEF = 4.8


def _logistic(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1.0 - p))


def logistic_winprob(state: GameState) -> float:
    """Closed-form fallback P(home wins). No training required."""
    secs = max(float(state.seconds_remaining), 0.0)
    # A lead's leverage grows as the clock shrinks (~1/sqrt(time)).
    lead_term = _LEAD_COEF * state.score_diff / math.sqrt(secs + 1.0)
    # The pre-game prior matters early and fades to nothing by the final buzzer.
    prior_weight = secs / 2880.0
    prior_term = _logit(state.pre_game_home_prob) * prior_weight
    return _logistic(lead_term + prior_term)


class WinProbModel:
    """Wraps an optional trained classifier; falls back to logistic if absent."""

    def __init__(self, estimator=None):
        self._estimator = estimator

    @classmethod
    def load(cls, path: str) -> "WinProbModel":
        p = Path(path)
        if not p.exists():
            return cls(None)
        import joblib  # local import: optional until a model exists

        return cls(joblib.load(p))

    @property
    def is_trained(self) -> bool:
        return self._estimator is not None

    def predict(self, state: GameState) -> float:
        """P(home team wins) in [0, 1]."""
        if self._estimator is None:
            return logistic_winprob(state)
        x = features_to_vector(state_to_features(state)).reshape(1, -1)
        return float(self._estimator.predict_proba(x)[0, 1])
