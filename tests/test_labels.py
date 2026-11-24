import numpy as np
import pandas as pd
import pytest

from jepa_cxr.chexpert import CHEXPERT_LABELS
from jepa_cxr.labels import (
    CHEXPERT_MAP,
    SHARED_LABELS,
    VINDR_MAP,
    frequent_labels,
    to_shared,
)

VINDR_COLS = [c for cols in VINDR_MAP.values() for c in cols] + ["COPD", "Lung Opacity"]


def test_maps_cover_shared_labels():
    assert set(CHEXPERT_MAP) == set(VINDR_MAP) == set(SHARED_LABELS)
    assert {c for cols in CHEXPERT_MAP.values() for c in cols} <= set(CHEXPERT_LABELS)


def test_chexpert_to_shared_keeps_values_and_nan():
    df = pd.DataFrame(0.0, index=["a", "b"], columns=CHEXPERT_LABELS)
    df.loc["a", "Edema"] = np.nan
    df.loc["b", "Support Devices"] = 1.0
    out = to_shared(df, CHEXPERT_MAP)
    assert list(out.columns) == list(SHARED_LABELS)
    assert np.isnan(out.loc["a", "Edema"])
    assert out.loc["b"].sum() == 0


def test_vindr_fracture_is_union():
    df = pd.DataFrame(0.0, index=["rib", "clav", "none"], columns=VINDR_COLS)
    df.loc["rib", "Rib fracture"] = 1.0
    df.loc["clav", "Clavicle fracture"] = 1.0
    df.loc["none", "Lung Opacity"] = 1.0
    out = to_shared(df, VINDR_MAP)
    assert out["Fracture"].tolist() == [1.0, 1.0, 0.0]
    assert out.loc["none"].sum() == 0


def test_union_nan_only_if_all_sources_missing():
    df = pd.DataFrame(
        {"Rib fracture": [np.nan, np.nan, 0.0], "Clavicle fracture": [1.0, np.nan, np.nan]}
    )
    out = to_shared(df, {name: ("Rib fracture", "Clavicle fracture") for name in SHARED_LABELS})
    assert out["Fracture"].iloc[0] == 1.0
    assert np.isnan(out["Fracture"].iloc[1])
    assert out["Fracture"].iloc[2] == 0.0


def test_missing_source_column_raises():
    with pytest.raises(ValueError, match="Nodule/Mass"):
        to_shared(pd.DataFrame(columns=["Atelectasis"]), VINDR_MAP)


def test_frequent_labels_needs_every_frame():
    a = pd.DataFrame({"x": [1, 1, 0], "y": [1, 1, 1], "z": [0, 0, 0]})
    b = pd.DataFrame({"x": [1, 1, 1], "y": [1, 0, np.nan], "z": [1, 1, 1]})
    assert frequent_labels([a, b], 2) == ["x"]
    assert frequent_labels([a], 2) == ["x", "y"]
