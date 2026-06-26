"""
AI-ENG-201 HW2 — Part 2.1
Ridge Regression via closed-form solution.
Author: Fatima Alibabayeva
"""

import numpy as np


class RidgeRegression:
    """Ridge regression using the closed-form solution.

    Solves w = (X^T X + λ I_p)^{-1} X^T y via np.linalg.solve.
    An intercept column is augmented into X; the intercept is NOT penalized
    (first diagonal entry of the penalty matrix is set to zero).
    No Python loop over training samples is used.
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        """Initialize Ridge regression.

        Parameters
        ----------
        lambda_ : float
            L2 regularization strength (λ ≥ 0).
        """
        self.lambda_ = lambda_
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        """Fit Ridge regression via the closed-form solution.

        Parameters
        ----------
        X : np.ndarray, shape (N, p)
        y : np.ndarray, shape (N,)

        Returns
        -------
        self
        """
        # intercept ucun 1-ler sutunu elave edirik
        X_ = np.hstack([np.ones((X.shape[0], 1)), X])

        # penalty matrixi: λ * I, amma intercept-i penalize etmirik
        I = np.eye(X_.shape[1])
        I[0, 0] = 0  # intercept-e penalty tətbiq olunmur (bias term-dir)

        # closed-form: (X^T X + λ I) w = X^T y
        A = X_.T @ X_ + self.lambda_ * I
        B = X_.T @ y
        self.w = np.linalg.solve(A, B)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute predictions for input X.

        Parameters
        ----------
        X : np.ndarray, shape (M, p)

        Returns
        -------
        np.ndarray, shape (M,)
        """
        X_ = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_ @ self.w
