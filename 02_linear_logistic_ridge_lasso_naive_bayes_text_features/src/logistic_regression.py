"""
AI-ENG-201 HW2 — Part 3.1 & 3.2
Binary Logistic Regression with L2 regularization via batch gradient descent.
Author: Fatima Alibabayeva
"""

import numpy as np


class LogisticRegression:
    """Binary logistic regression trained by gradient descent on cross-entropy.

    Loss (with L2 regularization):
        J(w) = -(1/N) sum_i [ y_i log σ(w^T x_i) + (1-y_i) log(1-σ(w^T x_i)) ]
               + (lambda_/(2N)) ||w||^2

    An intercept term is included by augmenting X with a column of ones.
    The intercept is NOT regularized (w[0] excluded from the penalty).
    Gradient is fully vectorized — no Python loop over samples.
    """

    def __init__(self, lr: float = 0.1, lambda_: float = 0.0, max_iter: int = 1000) -> None:
        """Initialize logistic regression.

        Parameters
        ----------
        lr       : learning rate for gradient descent.
        lambda_  : L2 regularization strength (0 = no regularization).
        max_iter : maximum gradient descent iterations.
        """
        self.lr = lr
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.w = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid: clips z to [-500, 500] before exp.

        Parameters
        ----------
        z : np.ndarray

        Returns
        -------
        np.ndarray  values in (0, 1)
        """
        # clip edirik ki overflow yaranmasin (exp(-500) ≈ 0, exp(500) → inf)
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fit logistic regression weights via gradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (N, p)
        y : np.ndarray, shape (N,)  — binary labels {0, 1}

        Returns
        -------
        self
        """
        # intercept ucun 1-ler sutunu elave edirik
        X_ = np.hstack([np.ones((X.shape[0], 1)), X])
        N, p = X_.shape
        self.w = np.zeros(p)

        for _ in range(self.max_iter):
            # predicted probability: σ(w^T x)
            y_pred = self._sigmoid(X_ @ self.w)

            # cross-entropy gradient: (1/N) X^T (y_pred - y)
            # sample-ler uzre loop YOX — matris əməliyyatı
            grad = (1 / N) * X_.T @ (y_pred - y)

            # L2 regularization gradient: (lambda/N) * w
            # intercept-i regularize etmirik (w[0] = bias term)
            reg = (self.lambda_ / N) * self.w.copy()
            reg[0] = 0  # intercept penalize edilmir
            grad = grad + reg

            self.w = self.w - self.lr * grad

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(y=1 | x) for each sample.

        Parameters
        ----------
        X : np.ndarray, shape (M, p)

        Returns
        -------
        np.ndarray, shape (M,)  — probabilities in [0, 1]
        """
        X_ = np.hstack([np.ones((X.shape[0], 1)), X])
        return self._sigmoid(X_ @ self.w)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels using threshold 0.5.

        Parameters
        ----------
        X : np.ndarray, shape (M, p)

        Returns
        -------
        np.ndarray, shape (M,)  — integer labels {0, 1}
        """
        # P >= 0.5 ise class 1, degilse class 0
        return (self.predict_proba(X) >= 0.5).astype(int)
