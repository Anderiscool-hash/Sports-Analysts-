"""Game-level split and model evaluation for the win-prob backtest.

Splits historical rows by game (never by row — every state in a game shares one
final label, so a row-level split would leak the answer), runs each test state
through model.predict, and scores with model/metrics.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sportedge.model.metrics import (
    CalibrationBin,
    accuracy,
    auc,
    brier_score,
    calibration_bins,
    log_loss,
)
from sportedge.types import GameState


@dataclass(frozen=True)
class BacktestReport:
    label: str
    n_games: int
    n_states: int
    brier: float
    log_loss: float
    accuracy: float
    auc: float
    calibration: list[CalibrationBin]


def _row_float(row: pd.Series, name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    if pd.isna(value):
        return default
    return float(value)


def split_by_game(
    df: pd.DataFrame,
    finals_game_ids: list[str] = (),
    test_frac: float = 0.3,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into (train, test) by game id. Finals ids are forced into test;
    the rest of the test set is a random sample sized to ~test_frac of all games."""
    game_ids = list(pd.unique(df["game_id"]))
    id_set = set(game_ids)
    finals = [g for g in finals_game_ids if g in id_set]
    finals_set = set(finals)
    others = [g for g in game_ids if g not in finals_set]

    rng = np.random.default_rng(seed)
    others_arr = np.array(others, dtype=object)
    rng.shuffle(others_arr)

    n_test_total = max(1, round(test_frac * len(game_ids)))
    n_from_others = max(0, n_test_total - len(finals))
    test_ids = finals_set | set(others_arr[:n_from_others].tolist())
    train_ids = id_set - test_ids

    assert train_ids.isdisjoint(test_ids), "game leaked across split"

    train_df = df[df["game_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["game_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df


def evaluate(model, rows: pd.DataFrame, label: str) -> BacktestReport:
    """Score `model` over `rows` (TRAINING_COLUMNS schema) and return a report."""
    if len(rows) == 0:
        return BacktestReport(
            label, 0, 0, float("nan"), float("nan"), float("nan"), float("nan"), []
        )
    labels = rows["home_win"].to_numpy(dtype=float)
    probs = np.empty(len(rows), dtype=float)
    for i, (_, r) in enumerate(rows.iterrows()):
        # Team names are irrelevant; optional rolling team-strength columns are
        # consumed when the cache has been enriched with NocturneBear totals.
        state = GameState(
            home_team="HOME",
            away_team="AWAY",
            home_score=int(r["home_score"]),
            away_score=int(r["away_score"]),
            period=int(r["period"]),
            seconds_remaining=float(r["seconds_remaining"]),
            pre_game_home_prob=float(r["pre_game_home_prob"]),
            home_recent_net_rating=_row_float(r, "home_recent_net_rating"),
            away_recent_net_rating=_row_float(r, "away_recent_net_rating"),
        )
        probs[i] = model.predict(state)
    return BacktestReport(
        label=label,
        n_games=int(rows["game_id"].nunique()),
        n_states=int(len(rows)),
        brier=brier_score(probs, labels),
        log_loss=log_loss(probs, labels),
        accuracy=accuracy(probs, labels),
        auc=auc(probs, labels),
        calibration=calibration_bins(probs, labels),
    )


def evaluate_overall_and_subset(
    model, test_df: pd.DataFrame, finals_game_ids: list[str] = ()
) -> dict[str, BacktestReport]:
    """Report on the whole test set, plus the Finals subset if any of its ids are
    present in `test_df`."""
    reports = {"overall": evaluate(model, test_df, "overall")}
    finals_rows = test_df[test_df["game_id"].isin(set(finals_game_ids))]
    if len(finals_rows) > 0:
        reports["finals"] = evaluate(model, finals_rows, "finals")
    return reports
