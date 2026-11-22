"""Baseline classifier: multi-label logistic regression on frozen embeddings."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, log_expit
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class LogisticHead:
    W: np.ndarray
    b: np.ndarray
    mean: np.ndarray
    std: np.ndarray  
    temperature: float = 1.0

    def logits(self, X: np.ndarray) -> np.ndarray:
        """Raw logits (N, L), before temperature scaling."""
        return ((X - self.mean) / self.std) @ self.W + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated probabilities (N, L)."""
        return expit(self.logits(X) / self.temperature)


def _bce(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-entry binary cross-entropy from logits, numerically stable."""
    return -(y * log_expit(z) + (1 - y) * log_expit(-z))


def _fit_one(Z: np.ndarray, y: np.ndarray, l2: float, max_iter: int) -> tuple[np.ndarray, float]:
    """Fit one binary logistic regression with scikit-learn."""
    if len(np.unique(y)) < 2:  # only one class seen: predict its rate, no weights
        rate = np.clip(y.mean(), 1e-3, 1 - 1e-3)
        return np.zeros(Z.shape[1]), float(np.log(rate / (1 - rate)))
    clf = LogisticRegression(C=1.0 / (l2 * len(y)), max_iter=max_iter)
    clf.fit(Z, y)
    return clf.coef_[0], float(clf.intercept_[0])


def fit_logistic(
    X: np.ndarray, Y: np.ndarray, l2: float = 1e-3, max_iter: int = 1000
) -> LogisticHead:
    """Fit one L2-regularized logistic regression per label."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    Z = (X - mean) / std

    W, b = np.zeros((X.shape[1], Y.shape[1])), np.zeros(Y.shape[1])
    for j in range(Y.shape[1]):
        keep = ~np.isnan(Y[:, j])
        W[:, j], b[j] = _fit_one(Z[keep], Y[keep, j], l2, max_iter)
    return LogisticHead(W=W, b=b, mean=mean, std=std)


def nll(head: LogisticHead, X: np.ndarray, Y: np.ndarray) -> float:
    """Mean binary cross-entropy over non-missing labels (with the head's temperature)."""
    Y = np.asarray(Y, dtype=float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    z = head.logits(X) / head.temperature
    keep = ~np.isnan(Y)
    return float(_bce(z[keep], Y[keep]).mean())


def select_l2(X_train, Y_train, X_val, Y_val, grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)) -> float:
    """Pick the L2 strength with the lowest validation NLL."""
    scores = {l2: nll(fit_logistic(X_train, Y_train, l2), X_val, Y_val) for l2 in grid}
    return min(scores, key=scores.get)


def fit_temperature(head: LogisticHead, X_val: np.ndarray, Y_val: np.ndarray) -> LogisticHead:
    """Fit one shared temperature on validation data by minimizing NLL."""
    Y = np.asarray(Y_val, dtype=float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    z = head.logits(X_val)
    keep = ~np.isnan(Y)

    def objective(log_t):
        return _bce(z[keep] / np.exp(log_t), Y[keep]).mean()

    res = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
    return replace(head, temperature=float(np.exp(res.x)))
