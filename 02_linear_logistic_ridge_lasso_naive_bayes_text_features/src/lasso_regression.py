"""
AI-ENG-201 HW2 — Part 2.2
Lasso Regression via cyclic coordinate descent with soft-thresholding.
Author: Fatima Alibabayeva
"""

import numpy as np


class LassoRegression:
    """Lasso regression using cyclic coordinate descent.

    Each coordinate w_j is updated via the soft-thresholding operator:
        w_j <- S(rho_j / denom_j, lambda / denom_j)
        S(z, t) = sign(z) * max(|z| - t, 0)

    where rho_j = x_j^T r^{(-j)} is the partial residual inner product,
    denom_j = ||x_j||^2, and r^{(-j)} = y - X_{-j} w_{-j} is the residual
    excluding feature j.  The residual computation over samples is fully
    vectorized — no Python loop over i.

    Features are standardized inside fit() so the penalty treats all
    coordinates equally.  predict() undoes the standardization automatically.
    The intercept is updated each iteration without regularization.
    """

    def __init__(self, lambda_: float = 1.0, max_iter: int = 10000, tol: float = 1e-4) -> None:
        """Initialize Lasso regression.

        Parameters
        ----------
        lambda_  : L1 regularization strength.
        max_iter : maximum coordinate descent iterations.
        tol      : early stopping threshold on max |w_new - w_old|.
        """
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        # fit zamanı öğrənilmiş standardizasiya parametrləri
        self.X_mean = None
        self.X_std = None
        self.intercept_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegression":
        """Fit Lasso via cyclic coordinate descent.

        Parameters
        ----------
        X : np.ndarray, shape (N, p)
        y : np.ndarray, shape (N,)

        Returns
        -------
        self
        """
        # featurelari standardize edirik ki penalty hamiya bərabər tətbiq olunsun
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        self.X_std[self.X_std == 0] = 1  # sifira bolunmemek ucun

        X_std = (X - self.X_mean) / self.X_std

        N, p = X_std.shape
        self.w = np.zeros(p)

        # intercept ayrıca hesablayacayiq — penalize edilmir
        self.intercept_ = np.mean(y)

        for iteration in range(self.max_iter):
            w_old = self.w.copy()

            for j in range(p):
                # j-ci feature-i cixaraq qalan featurelarin predictionini tapiriq
                # y_hat_minus_j = X w - w[j]*x[:,j] + intercept
                y_hat_minus_j = X_std @ self.w - self.w[j] * X_std[:, j] + self.intercept_

                # partial residual
                r_j = y - y_hat_minus_j

                # rho_j = x_j^T r_j  — vectorized, sample-ler uzre loop YOX
                rho_j = X_std[:, j] @ r_j

                # denom = ||x_j||^2
                denom = X_std[:, j] @ X_std[:, j]

                # soft-thresholding: S(z, lambda) = sign(z) * max(|z| - lambda, 0)
                z = rho_j / denom
                threshold = self.lambda_ / denom
                self.w[j] = np.sign(z) * max(abs(z) - threshold, 0)

            # intercept hər iterasiyada yenilenir (penalize edilmir)
            self.intercept_ = np.mean(y - X_std @ self.w)

            # early stopping: max weight dəyişimi tol-dan kicikdirsə dayanırıq
            if np.max(np.abs(self.w - w_old)) < self.tol:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute predictions for input X, undoing internal standardization.

        Parameters
        ----------
        X : np.ndarray, shape (M, p)

        Returns
        -------
        np.ndarray, shape (M,)
        """
        # predict etmeden evvel fit zamanı öğrənilmiş mean/std ile standardize edirik
        X_std = (X - self.X_mean) / self.X_std
        return X_std @ self.w + self.intercept_
