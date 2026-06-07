"""Pure scoring metrics for probabilistic forecasts.

Dependency-light (numpy only) and fully unit-tested, mirroring market/edge.py.
Each function takes parallel 1-D arrays: probs (model P in [0, 1]) and labels (0/1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    y = np.asarray(labels, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def accuracy(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    preds = (p >= threshold).astype(float)
    return float(np.mean(preds == y))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties resolved to their average (for rank-based AUC)."""
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(len(values), dtype=float)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # ranks are 1-based
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U). nan if only one class is present."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(p)
    sum_pos = float(np.sum(ranks[y == 1]))
    return (sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


@dataclass(frozen=True)
class CalibrationBin:
    lo: float
    hi: float
    count: int
    mean_pred: float
    observed_freq: float


def calibration_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> list[CalibrationBin]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    for k in range(n_bins):
        lo, hi = float(edges[k]), float(edges[k + 1])
        if k == n_bins - 1:  # include the right edge in the final bin
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(np.sum(mask))
        if count == 0:
            bins.append(CalibrationBin(lo, hi, 0, float("nan"), float("nan")))
        else:
            bins.append(
                CalibrationBin(
                    lo, hi, count, float(np.mean(p[mask])), float(np.mean(y[mask]))
                )
            )
    return bins
