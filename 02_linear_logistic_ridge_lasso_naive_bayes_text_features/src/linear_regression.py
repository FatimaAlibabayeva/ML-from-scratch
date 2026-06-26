"""
AI-ENG-201 HW2 — Part 1.1
OLS Linear Regression via normal equations.
Author: Fatima Alibabayeva
"""

import numpy as np


class LinearRegression:
    """Ordinary Least Squares linear regression using the normal equations.

    Solves w = (X^T X)^{-1} X^T y via np.linalg.solve for numerical stability.
    An intercept term is included by augmenting X with a column of ones.
    No Python loop over training samples is used anywhere in fit or predict.
    """

    def __init__(self) -> None:
        """Initialize the model; weights are set after calling fit."""
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Fit the model using the normal equations.

        Parameters
        ----------
        X : np.ndarray, shape (N, p)
            Design matrix of training features.
        y : np.ndarray, shape (N,)
            Target vector.

        Returns
        -------
        self
        """
        # intercept ucun 1-ler sutunu elave edirik
        # X_ = [[1, x1, x2, ...], [1, x1, x2, ...], ...]
        X_ = np.hstack([np.ones((X.shape[0], 1)), X])

        # normal equations: (X^T X) w = X^T y
        # solve() istifade edirik — inv()-den daha stabil, LU faktorizasiya ishledir
        A = X_.T @ X_
        b = X_.T @ y
        self.w = np.linalg.solve(A, b)
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
