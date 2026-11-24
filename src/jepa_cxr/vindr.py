"""VinDr-CXR image-level labels and DICOM decoding (policy in docs/data.md)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.pixels import apply_voi_lut

ID_COLS = ("image_id", "rad_id")
AGGREGATIONS = ("majority", "any")


def read_vindr_labels(path: str | Path, aggregate: str = "majority") -> pd.DataFrame:
    """One row per image, indexed by image_id, with radiologist reads combined."""
    if aggregate not in AGGREGATIONS:
        raise ValueError(f"aggregate must be one of {AGGREGATIONS}, got {aggregate!r}")
    raw = pd.read_csv(path)
    label_cols = [c for c in raw.columns if c not in ID_COLS]
    values = raw[label_cols].astype(float)
    unexpected = set(np.unique(values.to_numpy())) - {0.0, 1.0}
    if unexpected:
        raise ValueError(f"Unexpected label values in {path}: {sorted(unexpected)}")

    grouped = values.groupby(raw["image_id"])
    labels = grouped.mean() > 0.5 if aggregate == "majority" else grouped.max() > 0
    return labels.astype(float).rename_axis("image_id")


def read_dicom(path: str | Path) -> np.ndarray:
    """Decode a DICOM to uint8 MONOCHROME2 at full resolution, applying its VOI LUT/window."""
    ds = pydicom.dcmread(path)
    x = apply_voi_lut(ds.pixel_array, ds).astype(np.float64)
    if ds.PhotometricInterpretation == "MONOCHROME1":
        x = x.max() - x
    lo, hi = x.min(), x.max()
    x = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
    return (x * 255).round().astype(np.uint8)
