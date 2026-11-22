from abc import ABC, abstractmethod

import numpy as np

from jepa_cxr.classifiers import fit_logistic, fit_temperature, select_l2
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


METHODS: dict[str, type[UncertaintyMethod]] = {
    SoftmaxMethod.name: SoftmaxMethod,
}
