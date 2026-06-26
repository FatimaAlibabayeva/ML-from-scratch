"""
AI-ENG-201 HW2 — Part 1.2
Linear Regression via batch gradient descent on MSE loss.
Author: Fatima Alibabayeva
"""

import numpy as np


class LinearRegressionGD:
    """Batch gradient descent linear regression.

    Minimizes MSE loss J(w) = (1/N) * ||y - X w||^2 using vectorized gradients.
    Intercept is included by augmenting X with a bias column of ones.
    Stopping criterion: ||w_new - w_old||_2 < tol  OR  max_iter reached.
    """

    def __init__(self, lr: float = 0.01, max_iter: int = 1000, tol: float = 1e-6) -> None:
        """Initialize gradient descent hyperparameters.

        Parameters
        ----------
        lr       : learning rate η
        max_iter : maximum number of gradient steps
        tol      : convergence threshold on weight change ||w_new - w_old||
        """
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.mse_history = []  # hər iterasiyada MSE-ni saxlayiriq (plot ucun)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        """Fit weights via batch gradient descent.

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
        N = X_.shape[0]  # sample sayi

        w = np.zeros(X_.shape[1])  # sifirdan bashlayiriq
        self.mse_history = []

        for _ in range(self.max_iter):
            y_pred = X_ @ w

            # MSE-ni yadda saxlayiriq
            mse = np.mean((y - y_pred) ** 2)
            self.mse_history.append(mse)

            # vectorized gradient: -2/N * X^T (y - Xw)
            # sample-ler uzre loop YOX — matris əməliyyatı
            gradient = (-2 / N) * X_.T @ (y - y_pred)

            w_new = w - self.lr * gradient

            # convergence yoxlaması: weight dəyişimi kicikdirsə dayanırıq
            if np.linalg.norm(w_new - w) < self.tol:
                w = w_new
                self.mse_history.append(np.mean((y - X_ @ w) ** 2))
                break

            w = w_new

        self.w = w
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
