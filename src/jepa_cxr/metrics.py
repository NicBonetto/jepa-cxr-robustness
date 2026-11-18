"""Model-agnostic evaluation metrics: performance, calibration, and failure detection."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

EPS = 1e-12


def _2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[:, None] if a.ndim == 1 else a

def performance(y_true, y_prob) -> dict:
    """Per-label and macro AUROC / AUPRC. Labels with only one class present get NaN."""
    y_true, y_prob = _2d(y_true), _2d(y_prob)
    auroc, auprc = [], []
    for j in range(y_true.shape[1]):
        if len(np.unique(y_true[:, j])) < 2:
            auroc.append(np.nan)
            auprc.append(np.nan)
            continue
        auroc.append(roc_auc_score(y_true[:, j], y_prob[:, j]))
        auprc.append(average_precision_score(y_true[:, j], y_prob[:, j]))
    return {
        "auroc": np.array(auroc),
        "auprc": np.array(auprc),
        "macro_auroc": float(np.nanmean(auroc)),
        "macro_auprc": float(np.nanmean(auprc)),
    }


def ece(y_true, y_prob, n_bins: int = 15) -> float:
    """Expected calibration error over all (image, label) probabilities.

    Bins predicted probabilities into equal-width bins and averages |mean predicted -
    observed frequency|, weighted by bin size. 0 = perfectly calibrated.
    """
    y, p = _2d(y_true).ravel(), _2d(y_prob).ravel()
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.sum() * abs(p[mask].mean() - y[mask].mean())
    return float(total / len(p))

def entropy(y_prob) -> np.ndarray:
    """Binary entropy per (image, label), in nats. Max ln(2) at p = 0.5."""
    p = np.clip(np.asarray(y_prob, dtype=float), EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def knn_distance(train_emb, test_emb, k: int = 10, metric: str = "euclidean") -> np.ndarray:
    """Mean distance from each test embedding to its k nearest training embeddings."""
    nn = NearestNeighbors(n_neighbors=k, metric=metric).fit(train_emb)
    dist, _ = nn.kneighbors(test_emb)
    return dist.mean(axis=1)


def image_uncertainty(per_label, reduce: str = "max") -> np.ndarray:
    """Collapse (N, L) uncertainty to one score per image ('max' or 'mean')."""
    u = _2d(per_label)
    return u.max(axis=1) if reduce == "max" else u.mean(axis=1)


def image_errors(y_true, y_prob, threshold=0.5) -> np.ndarray:
    """Fraction of labels misclassified per image. `threshold` may be per-label (L,)."""
    y_true, y_prob = _2d(y_true), _2d(y_prob)
    y_pred = y_prob >= np.asarray(threshold)
    return (y_pred != y_true.astype(bool)).mean(axis=1)

def risk_coverage(errors, uncertainty) -> tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve.

    Keeps cases from most to least confident. At coverage k/N, risk is the mean error of
    the k cases kept. `errors` can be 0/1 or a per-case loss in [0, 1].
    """
    errors = np.asarray(errors, dtype=float)
    order = np.argsort(uncertainty, kind="stable")  # most confident first
    kept = np.arange(1, len(errors) + 1)
    risk = np.cumsum(errors[order]) / kept
    return kept / len(errors), risk


def aurc(errors, uncertainty) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    _, risk = risk_coverage(errors, uncertainty)
    return float(risk.mean())


def error_detection_auroc(errors, uncertainty) -> float:
    """AUROC of uncertainty for separating wrong cases (errors > 0) from right ones.

    1.0 = uncertainty ranks every error above every correct case
    0.5 = no better than chance.
    """
    wrong = np.asarray(errors) > 0
    if wrong.all() or not wrong.any():
        return float("nan")
    return float(roc_auc_score(wrong, uncertainty))


def evaluate(y_true, y_prob, uncertainty, threshold=0.5, n_bins: int = 15) -> dict:
    """Run all metrics. `uncertainty` is one score per image (see image_uncertainty)."""
    errors = image_errors(y_true, y_prob, threshold)
    return {
        **performance(y_true, y_prob),
        "ece": ece(y_true, y_prob, n_bins),
        "aurc": aurc(errors, uncertainty),
        "error_detection_auroc": error_detection_auroc(errors, uncertainty),
        "mean_error": float(errors.mean()),
    }

def plot_reliability(y_true, y_prob, n_bins: int = 15, ax=None, label=None):
    """Reliability diagram: observed frequency vs. mean predicted probability per bin."""
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    y, p = _2d(y_true).ravel(), _2d(y_prob).ravel()
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    used = [b for b in range(n_bins) if (bins == b).any()]
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.plot(
        [p[bins == b].mean() for b in used],
        [y[bins == b].mean() for b in used],
        marker="o",
        label=label,
    )
    ax.set(
        xlabel="Mean predicted probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1)
    )
    return ax

def plot_referral(errors, uncertainty, ax=None, label=None):
    """Referral curve: accuracy on retained cases vs. % of cases deferred to a human."""
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    coverage, risk = risk_coverage(errors, uncertainty)
    ax.plot(100 * (1 - coverage), 1 - risk, label=label)
    ax.set(xlabel="% of cases referred", ylabel="Accuracy on retained cases", xlim=(0, 100))
    return ax

