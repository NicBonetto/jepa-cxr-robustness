"""Shared CheXpert/VinDr label set (choices in docs/data.md)."""

import pandas as pd

SHARED_LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
    "Pneumothorax",
    "Pneumonia",
    "Lung Lesion",
    "Fracture",
    "No Finding",
)

CHEXPERT_MAP = {label: (label,) for label in SHARED_LABELS}

VINDR_MAP = {
    "Atelectasis": ("Atelectasis",),
    "Cardiomegaly": ("Cardiomegaly",),
    "Consolidation": ("Consolidation",),
    "Edema": ("Edema",),
    "Pleural Effusion": ("Pleural effusion",),
    "Pneumothorax": ("Pneumothorax",),
    "Pneumonia": ("Pneumonia",),
    "Lung Lesion": ("Nodule/Mass",),
    "Fracture": ("Rib fracture", "Clavicle fracture"),
    "No Finding": ("No finding",),
}


def to_shared(labels: pd.DataFrame, mapping: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    """Map dataset labels onto SHARED_LABELS, OR-ing multiple sources (NaN only if all are NaN)."""
    missing = sorted({c for cols in mapping.values() for c in cols} - set(labels.columns))
    if missing:
        raise ValueError(f"Labels missing source columns: {missing}")
    return pd.DataFrame(
        {name: labels[list(mapping[name])].max(axis=1) for name in SHARED_LABELS},
        index=labels.index,
    )


def frequent_labels(frames: list[pd.DataFrame], min_positives: int) -> list[str]:
    """Labels with at least `min_positives` positives in every frame, in column order."""
    cols = list(frames[0].columns)
    return [c for c in cols if all((f[c] == 1).sum() >= min_positives for f in frames)]
