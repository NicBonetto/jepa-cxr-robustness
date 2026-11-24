from abc import ABC, abstractmethod

import numpy as np

from jepa_cxr.classifiers import fit_logistic, fit_temperature, select_l2
from jepa_cxr.laplace import fit_laplace
from jepa_cxr.metrics import entropy, image_uncertainty


class UncertaintyMethod(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X_train, Y_train, X_val, Y_val) -> "UncertaintyMethod":
        """Train on (X_train, Y_train); use (X_val, Y_val) for any tuning."""

    @abstractmethod
    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Return (probs (N, L), per-image uncertainty (N,))."""


class SoftmaxMethod(UncertaintyMethod):
    """Logistic regression + temperature scaling; uncertainty = predictive entropy."""

    name = "softmax"

    def __init__(self, l2_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), reduce: str = "max"):
        self.l2_grid = l2_grid
        self.reduce = reduce  # how per-label uncertainty becomes one score per image

    def fit(self, X_train, Y_train, X_val, Y_val) -> "SoftmaxMethod":
        self.l2 = select_l2(X_train, Y_train, X_val, Y_val, self.l2_grid)
        head = fit_logistic(X_train, Y_train, self.l2)
        self.head = fit_temperature(head, X_val, Y_val)
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        probs = self.head.predict_proba(X)
        return probs, image_uncertainty(entropy(probs), self.reduce)

class LaplaceMethod(UncertaintyMethod):
    """Last-layer Laplace on the logistic regression head; uncertainty = predictive entropy."""

    name = "laplace"

    def __init__(
        self,
        l2_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0),
        prior_scales=(1.0,),
        reduce: str = "max",
    ):
        self.l2_grid = l2_grid
        self.prior_scales = prior_scales
        self.reduce = reduce

    def fit(self, X_train, Y_train, X_val, Y_val) -> "LaplaceMethod":
        self.l2 = select_l2(X_train, Y_train, X_val, Y_val, self.l2_grid)
        head = fit_logistic(X_train, Y_train, self.l2)
        best = None
        for scale in self.prior_scales:
            post = fit_laplace(head, X_train, Y_train, self.l2, scale)
            score = _nll(post.predict_proba(X_val), Y_val)
            if best is None or score < best[0]:
                best = (score, scale, post)
        _, self.prior_scale, self.posterior = best
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        probs = self.posterior.predict_proba(X)
        return probs, image_uncertainty(entropy(probs), self.reduce)


class LaplaceTunedMethod(LaplaceMethod):
    """Laplace with the prior precision retuned on validation NLL."""

    name = "laplace_tuned"

    def __init__(self, **kwargs):
        kwargs.setdefault("prior_scales", (0.01, 0.1, 1.0, 10.0, 100.0))
        super().__init__(**kwargs)

def _nll(probs: np.ndarray, Y) -> float:
    Y = np.asarray(Y, dtype=float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    p = np.clip(probs, 1e-12, 1 - 1e-12)
    keep = ~np.isnan(Y)
    return float(-(Y * np.log(p) + (1 - Y) * np.log(1 - p))[keep].mean())


METHODS: dict[str, type[UncertaintyMethod]] = {
    SoftmaxMethod.name: SoftmaxMethod,
    LaplaceMethod.name: LaplaceMethod,
    LaplaceTunedMethod.name: LaplaceTunedMethod
}
