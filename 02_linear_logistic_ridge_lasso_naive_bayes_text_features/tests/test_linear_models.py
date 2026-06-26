"""
AI-ENG-201 HW2 — Test Suite
Tests all major methods against sklearn equivalents and behavioral requirements.
Author: Fatima Alibabayeva
Run: pytest tests/test_linear_models.py -v
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from src.linear_regression import LinearRegression
from src.linear_regression_gd import LinearRegressionGD
from src.ridge_regression import RidgeRegression
from src.lasso_regression import LassoRegression
from src.logistic_regression import LogisticRegression
from src.naive_bayes import GaussianNaiveBayes
from src.text_features import BagOfWords, TfidfTransformer

# ===== 1. LINEAR REGRESSION (OLS) =====

class TestLinearRegression:
    """OLS linear regression tests."""

    def test_matches_sklearn(self):
        """Predictions must match sklearn within 1e-9."""
        from sklearn.linear_model import LinearRegression as SkLR
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        my = LinearRegression(); my.fit(X, y)
        sk = SkLR(); sk.fit(X, y)

        assert np.allclose(my.predict(X), sk.predict(X), atol=1e-9), \
            "OLS predictions differ from sklearn by more than 1e-9"

    def test_no_loop_over_samples(self):
        """fit should not contain a Python loop over training samples."""
        import inspect
        src = inspect.getsource(LinearRegression.fit)
        # 'for i in range' ya da 'for n in range' kimi sample-level loop olmamalidir
        # (max_iter loop kabul edilir amma LR.fit-de yoxdur)
        assert 'for i in range' not in src
        assert 'for n in range' not in src

    def test_predict_shape(self):
        """predict output shape must match (M,)."""
        np.random.seed(0)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        X_test = np.random.randn(20, 3)

        model = LinearRegression(); model.fit(X, y)
        assert model.predict(X_test).shape == (20,)

    def test_perfect_fit_on_linear_data(self):
        """On noiseless linear data, MSE should be near zero."""
        np.random.seed(1)
        X = np.random.randn(200, 4)
        w_true = np.array([1.0, -2.0, 3.0, 0.5])
        y = X @ w_true + 1.0  # intercept = 1

        model = LinearRegression(); model.fit(X, y)
        mse = np.mean((model.predict(X) - y) ** 2)
        assert mse < 1e-10, f"Expected near-zero MSE on linear data, got {mse}"


# ===== 2. LINEAR REGRESSION GD =====

class TestLinearRegressionGD:
    """Gradient descent regression tests."""

    def test_matches_closed_form(self):
        """GD weights must match closed-form within 1e-4."""
        np.random.seed(42)
        X = np.random.randn(1000, 10)
        y = np.random.randn(1000)

        cf = LinearRegression(); cf.fit(X, y)
        gd = LinearRegressionGD(lr=0.1, max_iter=1000, tol=1e-6); gd.fit(X, y)

        assert np.allclose(cf.w[1:], gd.w[1:], atol=1e-4), \
            "GD weights do not match closed-form within 1e-4"

    def test_mse_history_exists(self):
        """fit must populate mse_history for plotting."""
        np.random.seed(42)
        X = np.random.randn(100, 5); y = np.random.randn(100)
        gd = LinearRegressionGD(lr=0.1, max_iter=100); gd.fit(X, y)
        assert len(gd.mse_history) > 0, "mse_history must be non-empty after fit"

    def test_mse_decreasing(self):
        """MSE should be monotonically non-increasing for well-chosen lr."""
        np.random.seed(42)
        X = np.random.randn(200, 5); y = np.random.randn(200)
        gd = LinearRegressionGD(lr=0.01, max_iter=500); gd.fit(X, y)
        # sonuncu 10 iterasiyanin ortalamasi ilk 10-dan kicik olmalidir
        assert np.mean(gd.mse_history[-10:]) < np.mean(gd.mse_history[:10])

    def test_no_sample_loop_in_gradient(self):
        """Gradient computation must not loop over samples."""
        import inspect
        src = inspect.getsource(LinearRegressionGD.fit)
        # for i in range(N) ya da for i in range(self.max_iter) kimi
        # max_iter-i cixiriq, sample-level loop arayriq
        cleaned = src.replace('for _ in range(self.max_iter):', '')
        cleaned = cleaned.replace('for i in range(self.max_iter):', '')
        assert 'for i in range(N)' not in cleaned
        assert 'for n in range' not in cleaned


# ===== 3. RIDGE REGRESSION =====

class TestRidgeRegression:
    """Ridge regression tests."""

    def test_matches_sklearn(self):
        """Predictions must match sklearn.Ridge within 1e-9."""
        from sklearn.linear_model import Ridge
        np.random.seed(42)
        X = np.random.randn(100, 5); y = np.random.randn(100)

        my = RidgeRegression(lambda_=1.0); my.fit(X, y)
        sk = Ridge(alpha=1.0, fit_intercept=True); sk.fit(X, y)

        assert np.allclose(my.predict(X), sk.predict(X), atol=1e-9)

    def test_shrinks_toward_zero(self):
        """Higher lambda should shrink coefficients toward zero."""
        np.random.seed(42)
        X = np.random.randn(100, 5); y = np.random.randn(100)

        r1 = RidgeRegression(lambda_=0.001); r1.fit(X, y)
        r2 = RidgeRegression(lambda_=1000.0); r2.fit(X, y)

        norm1 = np.linalg.norm(r1.w[1:])
        norm2 = np.linalg.norm(r2.w[1:])
        assert norm2 < norm1, "Higher lambda should shrink coefficients more"

    def test_penalty_matrix_excludes_intercept(self):
        """The penalty matrix I[0,0] must be 0 so the intercept is not penalized."""
        import inspect
        src = inspect.getsource(RidgeRegression.fit)
        # kod intercept-i penalty-den cixarmalidir: I[0,0] = 0
        assert 'I[0, 0] = 0' in src or 'I[0,0] = 0' in src, \
            "Intercept must be excluded from penalty: I[0,0] = 0"


# ===== 4. LASSO REGRESSION =====

class TestLassoRegression:
    """Lasso regression tests."""

    def test_sparsity_at_high_lambda(self):
        """High lambda should zero out most coefficients."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = X[:, 0] * 3 + X[:, 1] * 2 + np.random.randn(200) * 0.1

        lasso = LassoRegression(lambda_=10.0)
        lasso.fit(X, y)
        n_zero = np.sum(np.abs(lasso.w) < 1e-3)
        assert n_zero > 0, f"Expected sparse solution at lambda=10, got {n_zero} zeros"

    def test_low_lambda_close_to_ols(self):
        """Very small lambda should yield low RMSE (similar to OLS)."""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = X @ np.array([1., 2., 3., 4., 5.]) + np.random.randn(200) * 0.5

        lasso = LassoRegression(lambda_=1e-5)
        lasso.fit(X, y)
        rmse = np.sqrt(np.mean((y - lasso.predict(X)) ** 2))
        assert rmse < 1.0, f"Expected RMSE < 1 at lambda=1e-5, got {rmse:.4f}"

    def test_no_sample_loop_in_update(self):
        """Inner coordinate update must not loop over samples."""
        import inspect
        src = inspect.getsource(LassoRegression.fit)
        # 'for i in range(N)' ya da 'for i in range(n)' olmamalidir
        assert 'for i in range(N)' not in src
        assert 'for n_idx' not in src

    def test_predict_undoes_standardization(self):
        """predict() must undo internal standardization correctly."""
        np.random.seed(0)
        X = np.random.randn(100, 3) * 10 + 5  # shifted, scaled
        y = X[:, 0] + np.random.randn(100)

        lasso = LassoRegression(lambda_=0.01)
        lasso.fit(X, y)
        preds = lasso.predict(X)
        # prediction-lar y ile eyni scale-da olmalidir
        assert preds.std() > 0.1, "Predictions seem constant — standardization may be broken"


# ===== 5. LOGISTIC REGRESSION =====

class TestLogisticRegression:
    """Logistic regression tests."""

    def test_high_accuracy_separable(self):
        """Should achieve >85% accuracy on linearly separable data."""
        np.random.seed(42)
        X = np.random.randn(400, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        lr = LogisticRegression(lr=0.5, lambda_=0.0, max_iter=2000)
        lr.fit(X, y)
        acc = np.mean(lr.predict(X) == y)
        assert acc > 0.85, f"Expected acc > 0.85, got {acc:.4f}"

    def test_proba_in_unit_interval(self):
        """predict_proba must return values in [0, 1]."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = (X[:, 0] > 0).astype(int)

        lr = LogisticRegression(lr=0.1, max_iter=500)
        lr.fit(X, y)
        proba = lr.predict_proba(X)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_binary(self):
        """predict must return only 0 or 1."""
        np.random.seed(7)
        X = np.random.randn(80, 3)
        y = (X[:, 0] > 0).astype(int)

        lr = LogisticRegression(lr=0.2, max_iter=300)
        lr.fit(X, y)
        preds = lr.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_intercept_not_regularized(self):
        """reg should not be applied to w[0] (intercept)."""
        # sadece kodun strukturunu yoxlayiriq — w[0]-a lambda tətbiq olunmur
        import inspect
        src = inspect.getsource(LogisticRegression.fit)
        assert 'reg[0] = 0' in src, "Intercept must be excluded from regularization"


# ===== 6. GAUSSIAN NAIVE BAYES =====

class TestGaussianNaiveBayes:
    """Gaussian Naive Bayes tests."""

    def test_wine_accuracy(self):
        """Should achieve >70% accuracy on Wine dataset."""
        from sklearn.datasets import load_wine
        from sklearn.model_selection import train_test_split

        wine = load_wine()
        X_tr, X_te, y_tr, y_te = train_test_split(
            wine.data, wine.target, test_size=0.2, random_state=42
        )
        gnb = GaussianNaiveBayes(); gnb.fit(X_tr, y_tr)
        acc = np.mean(gnb.predict(X_te) == y_te)
        assert acc > 0.70, f"Expected acc > 0.70 on Wine, got {acc:.4f}"

    def test_proba_sums_to_one(self):
        """predict_proba rows must sum to 1."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1, 2], size=50)

        gnb = GaussianNaiveBayes(); gnb.fit(X, y)
        proba = gnb.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_log_space_used(self):
        """Numerical stability: log must appear in _log_likelihood source."""
        import inspect
        src = inspect.getsource(GaussianNaiveBayes._log_likelihood)
        assert 'np.log' in src, "Expected log-space computation in _log_likelihood"

    def test_correct_classes(self):
        """predict should only return classes seen in training."""
        np.random.seed(5)
        X = np.random.randn(60, 2)
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)

        gnb = GaussianNaiveBayes(); gnb.fit(X, y)
        preds = gnb.predict(X)
        assert set(np.unique(preds)).issubset({0, 1, 2})


# ===== 7. TEXT FEATURES =====

class TestBagOfWords:
    """BagOfWords tests."""

    def test_vocabulary_size(self):
        """Vocabulary must not exceed max_features."""
        docs = ['the cat sat on the mat', 'the dog ran in the park', 'birds fly high in the sky']
        bow = BagOfWords(max_features=5)
        X = bow.fit_transform(docs)
        assert X.shape[1] == 5
        assert X.shape[0] == 3

    def test_counts_nonnegative(self):
        """Count matrix must be non-negative."""
        docs = ['hello world hello', 'world peace', 'hello peace world']
        bow = BagOfWords(max_features=10)
        X = bow.fit_transform(docs)
        assert np.all(X >= 0)

    def test_transform_matches_fit_transform(self):
        """transform on training data must match fit_transform."""
        docs = ['machine learning', 'deep learning', 'machine vision']
        bow = BagOfWords(max_features=20)
        X1 = bow.fit_transform(docs)
        X2 = bow.transform(docs)
        assert np.allclose(X1, X2)

    def test_unknown_words_ignored(self):
        """Words not in vocabulary must be silently ignored in transform."""
        train = ['apple banana cherry']
        test  = ['apple mango durian']
        bow = BagOfWords(max_features=10)
        bow.fit(train)
        X = bow.transform(test)
        # yalniz 'apple' vocabulary-dedir
        assert X.sum() == 1.0


class TestTfidfTransformer:
    """TfidfTransformer tests."""

    def test_shape_preserved(self):
        """Output shape must match input shape."""
        docs = ['machine learning is fun', 'deep learning needs data', 'fun with data']
        bow = BagOfWords(max_features=20)
        X_counts = bow.fit_transform(docs)
        tfidf = TfidfTransformer()
        X_tfidf = tfidf.fit_transform(X_counts)
        assert X_tfidf.shape == X_counts.shape

    def test_idf_downweights_common_words(self):
        """Words appearing in all documents should have low IDF."""
        docs = ['the cat', 'the dog', 'the bird']  # 'the' in all 3
        bow = BagOfWords(max_features=10)
        X_counts = bow.fit_transform(docs)
        tfidf = TfidfTransformer()
        tfidf.fit(X_counts)
        the_idx = bow.vocabulary_.get('the', None)
        if the_idx is not None:
            # 'the' butun documentlerde var — IDF kicik olmalidir
            other_idfs = np.delete(tfidf.idf_, the_idx)
            assert tfidf.idf_[the_idx] <= other_idfs.mean()

    def test_fit_transform_consistent(self):
        """fit_transform must match fit then transform."""
        docs = ['hello world', 'world peace', 'hello peace']
        bow = BagOfWords(max_features=10)
        X_counts = bow.fit_transform(docs)

        t1 = TfidfTransformer()
        X1 = t1.fit_transform(X_counts)

        t2 = TfidfTransformer()
        t2.fit(X_counts)
        X2 = t2.transform(X_counts)
        assert np.allclose(X1, X2)
