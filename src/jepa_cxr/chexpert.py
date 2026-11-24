"""CheXpert label loading and patient-level splits (policy in docs/data.md)."""

from pathlib import Path

import numpy as np
import pandas as pd

CHEXPERT_LABELS = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)

UNCERTAIN_POLICIES = {"ignore": np.nan, "zeros": 0.0, "ones": 1.0}
BLANK_POLICIES = {"zeros": 0.0, "ignore": np.nan}


def image_id_from_path(path: str) -> str:
    """'.../patient00001/study1/view1_frontal.jpg' -> 'patient00001_study1_view1_frontal'."""
    parts = Path(path).with_suffix("").parts
    return "_".join(parts[-3:])


def read_chexpert_csv(
    path: str | Path,
    uncertain: str = "ignore",
    blank: str = "zeros",
    frontal_only: bool = True,
) -> pd.DataFrame:
    """One row per image, indexed by image_id: path, patient_id, labels (NaN = missing)."""
    if uncertain not in UNCERTAIN_POLICIES:
        raise ValueError(f"uncertain must be one of {list(UNCERTAIN_POLICIES)}, got {uncertain!r}")
    if blank not in BLANK_POLICIES:
        raise ValueError(f"blank must be one of {list(BLANK_POLICIES)}, got {blank!r}")

    raw = pd.read_csv(path)
    if frontal_only:
        raw = raw[raw["Frontal/Lateral"] == "Frontal"]

    labels = raw[list(CHEXPERT_LABELS)].astype(float)
    unexpected = set(np.unique(labels.to_numpy()[~np.isnan(labels.to_numpy())])) - {-1.0, 0.0, 1.0}
    if unexpected:
        raise ValueError(f"Unexpected label values in {path}: {sorted(unexpected)}")
    labels = labels.fillna(BLANK_POLICIES[blank]).replace(-1.0, UNCERTAIN_POLICIES[uncertain])

    out = pd.DataFrame(
        {
            "path": raw["Path"].to_numpy(),
            "patient_id": [Path(p).parts[-3] for p in raw["Path"]],
        },
        index=pd.Index([image_id_from_path(p) for p in raw["Path"]], name="image_id"),
    )
    out[list(CHEXPERT_LABELS)] = labels.to_numpy()
    if not out.index.is_unique:
        raise ValueError(f"Duplicate image ids in {path}")
    return out


def split_by_patient(patient_ids: pd.Series, fractions: dict[str, float], seed: int) -> pd.Series:
    """Seeded split name per row; each patient lands in exactly one split."""
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {fractions}")
    patients = np.sort(patient_ids.unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    bounds = np.round(np.cumsum(list(fractions.values())) * len(patients)).astype(int)
    assignment = {}
    start = 0
    for name, end in zip(fractions, bounds, strict=True):
        assignment.update(dict.fromkeys(patients[start:end], name))
        start = end
    return patient_ids.map(assignment).rename("split")
