import numpy as np
import pytest
from sklearn import metrics as sk

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import accuracy, precision, recall, f1_score, roc_auc

np.random.seed(42)
y_true  = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
y_pred  = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
y_score = np.array([0.9, 0.2, 0.8, 0.6, 0.3, 0.7, 0.1, 0.5, 0.85, 0.15])


def test_accuracy():
    """accuracy must match sklearn.accuracy_score exactly."""
    assert accuracy(y_true, y_pred) == sk.accuracy_score(y_true, y_pred)


def test_precision():
    """precision must match sklearn.precision_score exactly."""
    assert precision(y_true, y_pred) == sk.precision_score(y_true, y_pred)


def test_recall():
    """recall must match sklearn.recall_score exactly."""
    assert recall(y_true, y_pred) == sk.recall_score(y_true, y_pred)


def test_f1():
    """f1_score must agree with sklearn within 1e-9."""
    assert abs(f1_score(y_true, y_pred) - sk.f1_score(y_true, y_pred)) < 1e-9


def test_roc_auc():
    """roc_auc must agree with sklearn within 1e-9."""
    assert abs(roc_auc(y_true, y_score) - sk.roc_auc_score(y_true, y_score)) < 1e-9


def test_roc_auc_with_knn_discretized_scores():
    """roc_auc must be correct even when scores are from a discrete {0,1/k,...,1} grid (k=5)."""
    # simulate k=5 KNN proba output — only 6 distinct values
    rng = np.random.default_rng(0)
    n = 200
    y_t = rng.integers(0, 2, n)
    # discretize to k+1 levels
    raw = rng.random(n)
    y_s = np.round(raw * 5) / 5
    assert abs(roc_auc(y_t, y_s) - sk.roc_auc_score(y_t, y_s)) < 1e-9


def test_precision_all_negative_pred():
    """precision should return 0 when no positive predictions are made (avoid division by zero)."""
    y_t = np.array([1, 1, 0, 0])
    y_p = np.array([0, 0, 0, 0])
    assert precision(y_t, y_p) == 0.0


def test_recall_all_negative_true():
    """recall should return 0 when there are no actual positives."""
    y_t = np.array([0, 0, 0, 0])
    y_p = np.array([1, 0, 1, 0])
    assert recall(y_t, y_p) == 0.0
