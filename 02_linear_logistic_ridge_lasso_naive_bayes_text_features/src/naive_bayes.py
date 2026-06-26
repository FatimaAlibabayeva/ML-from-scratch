"""
AI-ENG-201 HW2 — Part 3.3
Gaussian Naive Bayes with log-space computation.
Author: Fatima Alibabayeva
"""

import numpy as np


class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier.

    For each class c, estimates:
        - Prior: π_c = N_c / N
        - Per-feature Gaussian: μ_{jc}, σ²_{jc}

    Prediction uses log-space to avoid numerical underflow:
        log P(y=c | x) ∝ log π_c + Σ_j log N(x_j; μ_{jc}, σ²_{jc})

    A small epsilon (1e-9) is added to variances for numerical stability.
    """

    def __init__(self) -> None:
        """Initialize; parameters are set after calling fit."""
        self.classes_ = None
        self.priors_ = None   # P(y=c) — hər class ucun prior, shape (K,)
        self.means_ = None    # μ_{jc} — hər class, hər feature ucun mean, shape (K, p)
        self.vars_ = None     # σ²_{jc} — hər class, hər feature ucun variance, shape (K, p)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        """Estimate class priors and per-class Gaussian parameters.

        Parameters
        ----------
        X : np.ndarray, shape (N, p)
        y : np.ndarray, shape (N,)  — integer class labels

        Returns
        -------
        self
        """
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        N, p = X.shape

        self.priors_ = np.zeros(K)
        self.means_ = np.zeros((K, p))
        self.vars_ = np.zeros((K, p))

        for i, c in enumerate(self.classes_):
            # c classına aid olan sampleleri gotururuk
            X_c = X[y == c]

            # prior: P(y=c) = N_c / N
            self.priors_[i] = X_c.shape[0] / N

            # hər feature ucun Gaussian parametrlerini hesablayiriq
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0)

        return self

    def _log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Compute log P(x | y=c) + log P(y=c) for all classes.

        Parameters
        ----------
        X : np.ndarray, shape (N, p)

        Returns
        -------
        np.ndarray, shape (N, K)  — log unnormalized posteriors
        """
        N = X.shape[0]
        K = len(self.classes_)
        log_liks = np.zeros((N, K))

        for i in range(K):
            log_prior = np.log(self.priors_[i])

            # sifir variance-dan qorunmaq ucun kicik epsilon elave edirik
            var = self.vars_[i] + 1e-9

            # log N(x_j; μ_jc, σ²_jc) = -0.5 log(2πσ²) - (x-μ)²/(2σ²)
            # axis=1 ile butun featureleri eyni anda sumluruq — sample-ler uzre loop YOX
            log_lik = -0.5 * np.sum(
                np.log(2 * np.pi * var) + (X - self.means_[i]) ** 2 / var,
                axis=1
            )
            log_liks[:, i] = log_prior + log_lik

        return log_liks

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels.

        Parameters
        ----------
        X : np.ndarray, shape (M, p)

        Returns
        -------
        np.ndarray, shape (M,)
        """
        # hər sample ucun en yuksek log posterior olan classi secir
        log_liks = self._log_likelihood(X)
        return self.classes_[np.argmax(log_liks, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return normalized class probabilities using the log-sum-exp trick.

        Parameters
        ----------
        X : np.ndarray, shape (M, p)

        Returns
        -------
        np.ndarray, shape (M, K)  — rows sum to 1
        """
        log_liks = self._log_likelihood(X)

        # log-sum-exp trick: max cixiriq ki overflow olmasin
        log_liks -= log_liks.max(axis=1, keepdims=True)
        probs = np.exp(log_liks)
        return probs / probs.sum(axis=1, keepdims=True)
