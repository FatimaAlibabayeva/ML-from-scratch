import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.knn import KNN


def test_knn_predictions_match_sklearn():
    """Sanity check: custom KNN predictions must agree with sklearn on every test point."""
    np.random.seed(42)  # c++ daki time(NULL()) kimidi
    x_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    x_test  = np.random.rand(20, 5)

    # menim knn
    my_knn = KNN(k=5, metric="euclidean")
    my_knn.fit(x_train, y_train)
    my_pred  = my_knn.predict(x_test)
    my_proba = my_knn.predict_proba(x_test)

    # sklearn knn i
    sk_knn = KNeighborsClassifier(n_neighbors=5)
    sk_knn.fit(x_train, y_train)
    sk_pred  = sk_knn.predict(x_test)
    sk_proba = sk_knn.predict_proba(x_test)

    # neticeler eynidise davam or error
    assert np.array_equal(my_pred, sk_pred), "predict() disagrees with sklearn"
    assert np.allclose(my_proba, sk_proba, atol=1e-9), "predict_proba() disagrees with sklearn"


def test_knn_manhattan():
    """Manhattan metric must also match sklearn (metric='manhattan')."""
    np.random.seed(42)
    X = np.random.rand(80, 4)
    y = np.random.randint(0, 2, 80)
    Xt = np.random.rand(15, 4)

    my_knn = KNN(k=5, metric="manhattan")
    my_knn.fit(X, y)

    sk_knn = KNeighborsClassifier(n_neighbors=5, metric="manhattan")
    sk_knn.fit(X, y)

    assert np.array_equal(my_knn.predict(Xt), sk_knn.predict(Xt))


def test_knn_minkowski_reduces_to_euclidean():
    """Minkowski with q=2 must give identical results to euclidean."""
    np.random.seed(42)
    X  = np.random.rand(60, 3)
    y  = np.random.randint(0, 2, 60)
    Xt = np.random.rand(10, 3)

    m_euclidean  = KNN(k=5, metric="euclidean").fit(X, y)
    m_minkowski2 = KNN(k=5, metric="minkowski", q=2.0).fit(X, y)

    assert np.array_equal(m_euclidean.predict(Xt), m_minkowski2.predict(Xt))
    assert np.allclose(
        m_euclidean.predict_proba(Xt),
        m_minkowski2.predict_proba(Xt),
        atol=1e-9,
    )


def test_distance_weights_equidistant_reduces_to_uniform():
    """
    Weighted KNN bonus: when all k neighbors are equidistant,
    distance weighting must produce the same prediction as uniform weighting.
    """
    # craft data so every training point is the same distance from the test point
    X_train = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    y_train = np.array([1, 1, 0])
    X_test  = np.array([[0.0, 0.0]])  # equidistant from all three

    m_uniform  = KNN(k=3, weights="uniform").fit(X_train, y_train)
    m_distance = KNN(k=3, weights="distance").fit(X_train, y_train)

    assert np.array_equal(m_uniform.predict(X_test), m_distance.predict(X_test)), (
        "Distance-weighted KNN should match uniform when all neighbors are equidistant"
    )


def test_predict_proba_sums_to_one():
    """predict_proba rows must sum to 1."""
    np.random.seed(7)
    X = np.random.rand(50, 4)
    y = np.random.randint(0, 2, 50)
    m = KNN(k=5).fit(X, y)
    proba = m.predict_proba(np.random.rand(10, 4))
    assert np.allclose(proba.sum(axis=1), 1.0), "predict_proba rows must sum to 1"


def test_fit_returns_self():
    """fit() must return self so chaining works."""
    knn = KNN(k=3)
    X = np.random.rand(30, 2)
    y = np.random.randint(0, 2, 30)
    result = knn.fit(X, y)
    assert result is knn
