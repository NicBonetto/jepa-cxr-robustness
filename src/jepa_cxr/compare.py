import numpy as np
import pandas as pd

from jepa_cxr.metrics import evaluate, image_brier, plot_referral
from jepa_cxr.uncertainty import UncertaintyMethod

Split = tuple[np.ndarray, np.ndarray]  # (X, Y)
SUMMARY_KEYS = ("macro_auroc", "macro_auprc", "ece", "aurc", "loss_spearman", "mean_brier")


def run_comparison(
    methods: list[UncertaintyMethod],
    train: Split,
    val: Split,
    tests: dict[str, Split],
) -> tuple[pd.DataFrame, dict]:
    """Fit each method once, then evaluate it on each test condition.

    Returns (results table, raw outputs). Raw outputs map (method, condition) to
    (probs, uncertainty), for plotting or further analysis.
    """
    rows, outputs = [], {}
    for method in methods:
        method.fit(*train, *val)
        for condition, (X, Y) in tests.items():
            probs, unc = method.predict(X)
            outputs[(method.name, condition)] = (probs, unc)
            res = evaluate(Y, probs, unc)
            rows.append(
                {"method": method.name, "condition": condition} | {k: res[k] for k in SUMMARY_KEYS}
            )
    return pd.DataFrame(rows), outputs


def plot_referral_grid(outputs: dict, tests: dict[str, Split]):
    """One referral-curve panel per test condition, one line per method."""
    import matplotlib.pyplot as plt

    conditions = list(tests)
    fig, axes = plt.subplots(
        1, len(conditions), figsize=(5 * len(conditions), 4), squeeze=False, sharey=True
    )
    for ax, condition in zip(axes[0], conditions, strict=True):
        Y = tests[condition][1]
        for (method, cond), (probs, unc) in outputs.items():
            if cond == condition:
                plot_referral(image_brier(Y, probs), unc, ax=ax, label=method)
        ax.set_title(condition)
        ax.legend()
    fig.tight_layout()
    return fig


def make_fake_splits(
    n_train: int = 2000,
    n_val: int = 500,
    n_test: int = 1000,
    dim: int = 64,
    n_labels: int = 5,
    shifts: tuple[float, ...] = (0.5, 1.0, 2.0),
    missing_frac: float = 0.1,
    seed: int = 0,
) -> tuple[Split, Split, dict[str, Split]]:
    """Synthetic embeddings with linear label structure."""
    rng = np.random.default_rng(seed)
    true_w = rng.normal(size=(dim, n_labels))
    bias = rng.normal(-1.0, 0.5, size=n_labels)  # mostly-negative labels, like rare findings

    def sample(n):
        X = rng.normal(size=(n, dim))
        logits = X @ true_w / np.sqrt(dim) * 3 + bias + rng.normal(scale=0.5, size=(n, n_labels))
        return X, (logits > 0).astype(float)

    train, val = sample(n_train), sample(n_val)
    Y_train = train[1].copy()
    Y_train[rng.random(Y_train.shape) < missing_frac] = np.nan

    X_test, Y_test = sample(n_test)
    tests = {"clean": (X_test, Y_test)}
    for s in shifts:
        tests[f"shift_{s:g}"] = (X_test + rng.normal(scale=s, size=X_test.shape), Y_test)
    return (train[0], Y_train), val, tests
