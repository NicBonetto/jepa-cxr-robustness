from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import expit

from jepa_cxr.classifiers import LogisticHead

BIAS_PRIOR = 1e-4


@dataclass(frozen=True)
class LaplacePosterior:
    head: LogisticHead
    chol: list[np.ndarray]

    def _features(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.head.mean) / self.head.std
        return np.hstack([Z, np.ones((len(Z), 1))])

    def logit_variance(self, X: np.ndarray) -> np.ndarray:
        """sigma^2 per (image, finding): how much plausible classifiers disagree."""
        Z1 = self._features(X)
        var = np.empty((len(X), len(self.chol)))
        for j, L in enumerate(self.chol):
            V = solve_triangular(L, Z1.T, lower=True)
            var[:, j] = (V**2).sum(axis=0)
        return var

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predictive probabilities (N, L) under the probit approximation."""
        mu = self.head.logits(X)
        return expit(mu / np.sqrt(1 + np.pi * self.logit_variance(X) / 8))


def fit_laplace(
    head: LogisticHead,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    l2: float,
    prior_scale: float = 1.0,
) -> LaplacePosterior:
    """Fit the Laplace posterior around an already-fitted (untempered) logistic head."""
    Y = np.asarray(Y_train, dtype=float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    Z = (np.asarray(X_train, dtype=float) - head.mean) / head.std
    Z1 = np.hstack([Z, np.ones((len(Z), 1))])
    p = expit(head.logits(X_train))

    chol = []
    for j in range(Y.shape[1]):
        keep = ~np.isnan(Y[:, j])
        prior = np.full(Z1.shape[1], prior_scale * l2 * keep.sum())
        prior[-1] = BIAS_PRIOR
        weights = p[keep, j] * (1 - p[keep, j])
        H = (Z1[keep].T * weights) @ Z1[keep] + np.diag(prior)
        chol.append(cholesky(H, lower=True))
    return LaplacePosterior(head=head, chol=chol)

