"""Feature engineering. Pure functions shared by training and live inference so
there is no train/serve skew."""

from __future__ import annotations

import numpy as np

from sportedge.types import GameState

FEATURE_NAMES = [
    "score_diff",
    "seconds_remaining",
    "diff_x_invtime",  # lead leverage: a lead matters more as the clock shrinks
    "pre_game_home_prob",
    "period",
]


def state_to_features(state: GameState) -> dict[str, float]:
    secs = max(float(state.seconds_remaining), 0.0)
    inv_time = 1.0 / (secs + 30.0)
    return {
        "score_diff": float(state.score_diff),
        "seconds_remaining": secs,
        "diff_x_invtime": float(state.score_diff) * inv_time,
        "pre_game_home_prob": float(state.pre_game_home_prob),
        "period": float(state.period),
    }


def features_to_vector(feats: dict[str, float]) -> np.ndarray:
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=float)
