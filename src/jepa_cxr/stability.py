"""Aim 1: how far corruptions move embeddings away from their clean versions."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from jepa_cxr.embeddings import load_embeddings
from jepa_cxr.extract import read_manifest

METRICS = ("cosine_mean", "cosine_median", "rel_shift_median", "left_neighbourhood", "knn_overlap")
GROUP = ["encoder", "dataset", "split"]


def align(
    ids_a: list[str], emb_a: np.ndarray, ids_b: list[str], emb_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder `emb_b` to match `ids_a`. Raises if the two id sets differ."""
    if len(set(ids_a)) != len(ids_a) or len(set(ids_b)) != len(ids_b):
        raise ValueError("Duplicate image ids in an embedding file")
    if set(ids_a) != set(ids_b):
        diff = set(ids_a) ^ set(ids_b)
        raise ValueError(f"Clean and corrupted image ids differ ({len(diff)} not shared)")
    pos = {i: n for n, i in enumerate(ids_b)}
    return emb_a, emb_b[[pos[i] for i in ids_a]]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine distance 1 - cos(a_i, b_i), in [0, 2]."""
    if a.shape != b.shape:
        raise ValueError(f"Shapes differ: {a.shape} vs {b.shape}")
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    if np.any(na == 0) or np.any(nb == 0):
        raise ValueError("Zero-norm embedding has no cosine distance")
    return 1 - np.einsum("ij,ij->i", a, b) / (na * nb)


def neighbours(emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and cosine distances of each row's k nearest other rows, nearest first."""
    if not 1 <= k < len(emb):
        raise ValueError(f"k must be in [1, {len(emb) - 1}], got {k}")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute").fit(emb)
    dist, idx = nn.kneighbors(emb)
    # Drop the query itself. With exact duplicates it may not be in column 0, or at all.
    keep = idx != np.arange(len(emb))[:, None]
    keep[keep.sum(axis=1) > k, -1] = False
    return idx[keep].reshape(len(emb), k), dist[keep].reshape(len(emb), k)


def knn_overlap(idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    """Per row, the fraction of the k neighbours in `idx_a` that also appear in `idx_b`."""
    if idx_a.shape != idx_b.shape:
        raise ValueError(f"Shapes differ: {idx_a.shape} vs {idx_b.shape}")
    k = idx_a.shape[1]
    return np.array([len(set(a) & set(b)) / k for a, b in zip(idx_a, idx_b, strict=True)])


def stability(clean: np.ndarray, corrupted: np.ndarray, k: int = 10, clean_nn=None) -> dict:
    """Summary of how far `corrupted` rows moved from the matching `clean` rows.

    - cosine_mean / cosine_median: cosine distance between each image's clean and corrupted
      embedding. Not comparable across encoders, whose embedding spaces differ in scale.
    - rel_shift_median: that distance divided by the image's clean distance to its k-th
      nearest clean neighbour. Scale-free; above 1 means the image moved further than its
      neighbourhood is wide.
    - left_neighbourhood: fraction of images with relative shift above 1.
    - knn_overlap: mean fraction of each image's k clean neighbours that are still among its
      k neighbours when every image is corrupted. 1 = neighbourhoods fully preserved.

    `clean_nn` is `neighbours(clean, k)`, passed in to reuse it across conditions.
    """
    idx_c, dist_c = clean_nn if clean_nn is not None else neighbours(clean, k)
    shift = cosine_distance(clean, corrupted)
    radius = dist_c[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(radius > 0, shift / radius, np.where(shift > 0, np.inf, 0.0))
    idx_x, _ = neighbours(corrupted, k)
    return {
        "n_images": len(clean),
        "cosine_mean": float(shift.mean()),
        "cosine_median": float(np.median(shift)),
        "cosine_std": float(shift.std()),
        "rel_shift_median": float(np.median(rel)),
        "left_neighbourhood": float(np.mean(rel > 1)),
        "knn_overlap": float(knn_overlap(idx_c, idx_x).mean()),
    }


def stability_table(
    manifest: str | Path,
    k: int = 10,
    encoders: list[str] | None = None,
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    """Stability of every corrupted condition in the manifest against its clean split.

    Rows are matched to the clean embeddings of the same encoder, dataset and split by image id.
    Conditions with no clean counterpart raise, rather than being skipped silently.
    """
    root = Path(manifest).parent
    m = read_manifest(manifest)
    if m.empty:
        raise ValueError(f"No embeddings recorded in {manifest}")
    if encoders:
        m = m[m["encoder"].isin(encoders)]
    if datasets:
        m = m[m["dataset"].isin(datasets)]

    rows = []
    for (enc, ds, split), g in m.groupby(GROUP, sort=True):
        corrupt_rows = g[g["corruption"] != "clean"]
        if corrupt_rows.empty:
            continue
        clean_rows = g[g["corruption"] == "clean"]
        if len(clean_rows) != 1:
            raise ValueError(
                f"Expected one clean file for {enc}/{ds}/{split}, got {len(clean_rows)}"
            )
        clean_row = clean_rows.iloc[0]
        ids_c, emb_c = load_embeddings(root / clean_row["path"])
        clean_nn = neighbours(emb_c, k)
        for _, r in corrupt_rows.sort_values(["corruption", "severity"]).iterrows():
            if r["seed"] != clean_row["seed"] or r["revision"] != clean_row["revision"]:
                raise ValueError(f"{r['path']}: seed or revision differs from its clean file")
            ids_x, emb_x = load_embeddings(root / r["path"])
            emb_c_al, emb_x_al = align(ids_c, emb_c, ids_x, emb_x)
            rows.append(
                {"encoder": enc, "dataset": ds, "split": split, "corruption": r["corruption"],
                 "severity": int(r["severity"]), "k": k}
                | stability(emb_c_al, emb_x_al, k, clean_nn)
            )  # fmt: skip
    if not rows:
        raise ValueError("No corrupted conditions with a clean counterpart in the manifest")
    return pd.DataFrame(rows)


def plot_severity(table: pd.DataFrame, metrics=("cosine_mean", "rel_shift_median", "knn_overlap")):
    """Metric vs severity: one row of panels per (encoder, dataset), one line per corruption.

    Severity 0 is the clean anchor (shift 0, overlap 1).
    """
    import matplotlib.pyplot as plt

    anchor = {"cosine_mean": 0.0, "cosine_median": 0.0, "rel_shift_median": 0.0,
              "left_neighbourhood": 0.0, "knn_overlap": 1.0}  # fmt: skip
    groups = list(table.groupby(["encoder", "dataset", "split"], sort=True))
    fig, axes = plt.subplots(
        len(groups), len(metrics), figsize=(4.5 * len(metrics), 3.5 * len(groups)), squeeze=False
    )
    for row, ((enc, ds, split), g) in zip(axes, groups, strict=True):
        for ax, metric in zip(row, metrics, strict=True):
            for corr, c in g.groupby("corruption", sort=True):
                c = c.sort_values("severity")
                ax.plot([0, *c["severity"]], [anchor[metric], *c[metric]], marker="o", label=corr)
            ax.set_title(f"{enc} · {ds}/{split}")
            ax.set_xlabel("severity")
            ax.set_ylabel(metric)
            ax.set_xticks(range(0, 6))
            if metric == "knn_overlap":
                ax.set_ylim(0, 1.02)
            if metric == "rel_shift_median":
                ax.axhline(1, color="grey", lw=0.8, ls="--")
    axes[0, -1].legend(fontsize=8)
    fig.tight_layout()
    return fig
