import math

import numpy as np

from sportedge.model.metrics import (
    CalibrationBin,
    accuracy,
    auc,
    brier_score,
    calibration_bins,
    log_loss,
)


def test_brier_perfect():
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    labels = np.array([1, 0, 1, 0])
    assert brier_score(probs, labels) == 0.0


def test_brier_all_half():
    probs = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 0, 1, 0])
    assert brier_score(probs, labels) == 0.25


def test_log_loss_perfect_near_zero():
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    labels = np.array([1, 0, 1, 0])
    assert log_loss(probs, labels) < 1e-10


def test_accuracy_half():
    probs = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 0, 1, 0])  # 2nd and 3rd predictions wrong
    assert accuracy(probs, labels) == 0.5


def test_auc_separable():
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert auc(probs, labels) == 1.0


def test_auc_single_class_is_nan():
    probs = np.array([0.3, 0.6, 0.9])
    labels = np.array([1, 1, 1])
    assert math.isnan(auc(probs, labels))


def test_auc_ties_half():
    probs = np.array([0.5, 0.5])
    labels = np.array([0, 1])
    assert auc(probs, labels) == 0.5


def test_calibration_bins_counts_and_freq():
    probs = np.array([0.05, 0.05, 0.95, 0.95])
    labels = np.array([0, 1, 1, 1])
    bins = calibration_bins(probs, labels, n_bins=10)
    assert isinstance(bins[0], CalibrationBin)
    assert bins[0].count == 2
    assert bins[0].mean_pred == 0.05
    assert bins[0].observed_freq == 0.5
    assert bins[9].count == 2
    assert bins[9].observed_freq == 1.0
    assert bins[5].count == 0
    assert math.isnan(bins[5].mean_pred)
