"""Model-agnostic evaluation metrics: performance, calibration, and failure detection."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

EPS = 1e-12


def _2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[:, None] if a.ndim == 1 else a

def performance(y_true, y_prob) -> dict:
    """Per-label and macro AUROC / AUPRC over observed labels. One-class labels get NaN."""
    y_true, y_prob = _2d(y_true), _2d(y_prob)
    auroc, auprc = [], []
    for j in range(y_true.shape[1]):
        keep = ~np.isnan(y_true[:, j])
        y, p = y_true[keep, j], y_prob[keep, j]
        if len(np.unique(y)) < 2:
            auroc.append(np.nan)
            auprc.append(np.nan)
            continue
        auroc.append(roc_auc_score(y, p))
        auprc.append(average_precision_score(y, p))
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
    y, p = y[~np.isnan(y)], p[~np.isnan(y)]
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


def image_brier(y_true, y_prob) -> np.ndarray:
    """Mean squared error between probability and label over each image's observed labels.

    In [0, 1]; NaN for an image with no observed labels.
    """
    y_true, y_prob = _2d(y_true), _2d(y_prob)
    observed = ~np.isnan(y_true)
    sq = np.where(observed, (y_prob - np.nan_to_num(y_true)) ** 2, 0.0)
    n = observed.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, sq.sum(axis=1) / n, np.nan)

def _observed(loss, uncertainty) -> tuple[np.ndarray, np.ndarray]:
    """Drop images whose loss is NaN (no observed labels)."""
    loss, uncertainty = np.asarray(loss, dtype=float), np.asarray(uncertainty)
    keep = ~np.isnan(loss)
    return loss[keep], uncertainty[keep]


def risk_coverage(loss, uncertainty) -> tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve.

    Keeps cases from most to least confident. At coverage k/N, risk is the mean loss of
    the k cases kept.
    """
    loss, uncertainty = _observed(loss, uncertainty)
    order = np.argsort(uncertainty, kind="stable")
    kept = np.arange(1, len(loss) + 1)
    risk = np.cumsum(loss[order]) / kept
    return kept / len(loss), risk


def aurc(loss, uncertainty) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    _, risk = risk_coverage(loss, uncertainty)
    return float(risk.mean())


def loss_spearman(loss, uncertainty) -> float:
    """Spearman correlation between uncertainty and per-image loss.

    1 = uncertainty ranks images exactly by loss, 0 = unrelated. NaN if either is constant.
    """
    loss, uncertainty = _observed(loss, uncertainty)
    if np.ptp(loss) == 0 or np.ptp(uncertainty) == 0:
        return float("nan")
    return float(spearmanr(loss, uncertainty).statistic)


def evaluate(y_true, y_prob, uncertainty, n_bins: int = 15) -> dict:
    """Run all metrics. `uncertainty` is one score per image (see image_uncertainty)."""
    loss = image_brier(y_true, y_prob)
    return {
        **performance(y_true, y_prob),
        "ece": ece(y_true, y_prob, n_bins),
        "aurc": aurc(loss, uncertainty),
        "loss_spearman": loss_spearman(loss, uncertainty),
        "mean_brier": float(np.nanmean(loss)),
    }

def plot_reliability(y_true, y_prob, n_bins: int = 15, ax=None, label=None):
    """Reliability diagram: observed frequency vs. mean predicted probability per bin."""
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    y, p = _2d(y_true).ravel(), _2d(y_prob).ravel()
    y, p = y[~np.isnan(y)], p[~np.isnan(y)]
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

def plot_referral(loss, uncertainty, ax=None, label=None):
    """Referral curve: mean Brier score on retained cases vs. % of cases referred."""
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    coverage, risk = risk_coverage(loss, uncertainty)
    ax.plot(100 * (1 - coverage), risk, label=label)
    ax.set(xlabel="% of cases referred", ylabel="Brier score on retained cases", xlim=(0, 100))
    return ax

