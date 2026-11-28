"""Aim 2: classifier performance and uncertainty across corrupted test conditions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from jepa_cxr.compare import SUMMARY_KEYS
from jepa_cxr.embeddings import load_embeddings
from jepa_cxr.extract import read_manifest
from jepa_cxr.metrics import _observed, evaluate, image_errors
from jepa_cxr.uncertainty import UncertaintyMethod


@dataclass(frozen=True)
class Condition:
    corruption: str
    severity: int
    path: Path

    @property
    def name(self) -> str:
        return "clean" if self.corruption == "clean" else f"{self.corruption}_s{self.severity}"


def find_files(
    manifest: str | Path, encoder: str, dataset: str, train: str, val: str, test: str
) -> tuple[Path, Path, list[Condition]]:
    """Clean train and val paths, and every test condition (clean first) for one encoder.

    Raises if a clean split is missing or recorded more than once, or if the files were made
    with different seeds or encoder revisions.
    """
    root = Path(manifest).parent
    m = read_manifest(manifest)
    m = m[(m["encoder"] == encoder) & (m["dataset"] == dataset)]
    if m.empty:
        raise ValueError(f"No {encoder}/{dataset} embeddings in {manifest}")

    def clean(split: str) -> Path:
        rows = m[(m["split"] == split) & (m["corruption"] == "clean")]
        if len(rows) != 1:
            raise ValueError(f"Expected one clean {encoder}/{dataset}/{split}, got {len(rows)}")
        return root / rows.iloc[0]["path"]

    train_path, val_path, test_clean = clean(train), clean(val), clean(test)
    used = m[m["split"].isin([train, val, test])]
    for col in ("seed", "revision"):
        if used[col].astype(str).nunique() > 1:
            raise ValueError(f"{encoder}/{dataset} files mix {col}s: {sorted(set(used[col]))}")

    corrupted = m[(m["split"] == test) & (m["corruption"] != "clean")]
    conditions = [Condition("clean", 0, test_clean)] + [
        Condition(r["corruption"], int(r["severity"]), root / r["path"])
        for _, r in corrupted.sort_values(["corruption", "severity"]).iterrows()
    ]
    return train_path, val_path, conditions


def load_split(path: str | Path, labels: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Embeddings and their label rows, in file order. Raises if any image has no labels."""
    ids, X = load_embeddings(path)
    missing = pd.Index(ids).difference(labels.index)
    if len(missing):
        raise ValueError(f"{len(missing)} images in {path} have no labels, e.g. {missing[0]}")
    return X, labels.loc[ids]


def referral_threshold(uncertainty, rate: float) -> float:
    """Uncertainty above which a fraction `rate` of cases is referred."""
    if not 0 < rate < 1:
        raise ValueError(f"rate must be in (0, 1), got {rate}")
    return float(np.quantile(np.asarray(uncertainty, dtype=float), 1 - rate))


def fixed_threshold_referral(errors, uncertainty, threshold: float) -> tuple[float, float]:
    """(fraction referred, mean error on the cases kept) for a fixed uncertainty threshold.

    Images with no observed labels are left out of both. Error is NaN if nothing is kept.
    """
    errors, uncertainty = _observed(errors, uncertainty)
    refer = uncertainty > threshold
    kept = errors[~refer]
    return float(refer.mean()), float(kept.mean()) if len(kept) else float("nan")


def run_sweep(
    methods: list[UncertaintyMethod],
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    conditions: list[Condition],
    load_test,
    referral_rates: tuple[float, ...] = (0.1, 0.2),
) -> pd.DataFrame:
    """Fit each method once on clean data, then evaluate it on every test condition.

    `load_test(condition)` returns (X, Y). Conditions are loaded one at a time so the whole
    grid never sits in memory. Referral thresholds are set on clean validation uncertainty
    (one per rate) and held fixed across conditions, as a deployed system would; the
    `referred@r` columns show how the referral rate drifts under corruption.
    """
    thresholds = {}
    for method in methods:
        method.fit(*train, *val)
        _, unc_val = method.predict(val[0])
        thresholds[method.name] = {r: referral_threshold(unc_val, r) for r in referral_rates}

    rows = []
    for cond in conditions:
        X, Y = load_test(cond)
        for method in methods:
            probs, unc = method.predict(X)
            res = evaluate(Y, probs, unc)
            row = {"method": method.name, "corruption": cond.corruption,
                   "severity": cond.severity, "n_images": len(X)}  # fmt: skip
            row |= {k: res[k] for k in SUMMARY_KEYS}
            errors = image_errors(Y, probs)
            for r, t in thresholds[method.name].items():
                row[f"referred@{r:g}"], row[f"kept_error@{r:g}"] = fixed_threshold_referral(
                    errors, unc, t
                )
            rows.append(row)
    return pd.DataFrame(rows)


PLOT_METRICS = ("macro_auroc", "ece", "aurc", "error_detection_auroc")


def plot_sweep(table: pd.DataFrame, metrics=PLOT_METRICS, title: str = ""):
    """One column per corruption, one row per metric, one line per method.

    The clean result is drawn at severity 0 on every panel.
    """
    import matplotlib.pyplot as plt

    clean = table[table["corruption"] == "clean"]
    corruptions = sorted(set(table["corruption"]) - {"clean"})
    fig, axes = plt.subplots(
        len(metrics), len(corruptions), squeeze=False, sharey="row",
        figsize=(3.2 * len(corruptions), 2.6 * len(metrics)),
    )  # fmt: skip
    for j, corr in enumerate(corruptions):
        sub = pd.concat([clean, table[table["corruption"] == corr]])
        for i, metric in enumerate(metrics):
            ax = axes[i, j]
            for method, g in sub.groupby("method", sort=True):
                g = g.sort_values("severity")
                ax.plot(g["severity"], g[metric], marker="o", ms=3, label=method)
            ax.set_xticks(range(0, 6))
            if i == 0:
                ax.set_title(corr)
            if i == len(metrics) - 1:
                ax.set_xlabel("severity")
            if j == 0:
                ax.set_ylabel(metric)
    axes[0, -1].legend(fontsize=7)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
