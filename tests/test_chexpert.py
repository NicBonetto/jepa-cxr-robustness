import numpy as np
import pandas as pd
import pytest
from PIL import Image

from jepa_cxr.chexpert import (
    CHEXPERT_LABELS,
    image_id_from_path,
    read_chexpert_csv,
    split_by_patient,
)
from jepa_cxr.images import load_batch, load_gray, save_png

ROOT = "CheXpert-v1.0-small/train"


def _row(patient: int, study: int, view: str, frontal: bool, **labels) -> dict:
    row = {
        "Path": f"{ROOT}/patient{patient:05d}/study{study}/{view}.jpg",
        "Sex": "Female",
        "Age": 60,
        "Frontal/Lateral": "Frontal" if frontal else "Lateral",
        "AP/PA": "AP" if frontal else "",
    }
    row.update(dict.fromkeys(CHEXPERT_LABELS, np.nan))
    row.update({k.replace("_", " "): v for k, v in labels.items()})
    return row


@pytest.fixture
def csv_path(tmp_path):
    rows = [
        _row(1, 1, "view1_frontal", True, Cardiomegaly=1.0, Edema=-1.0),
        _row(1, 1, "view2_lateral", False, Cardiomegaly=1.0),
        _row(2, 1, "view1_frontal", True, Cardiomegaly=0.0, Edema=1.0),
        _row(3, 2, "view1_frontal", True, Pleural_Effusion=-1.0),
    ]
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_image_id_is_unique_per_patient():
    assert image_id_from_path(f"{ROOT}/patient00001/study1/view1_frontal.jpg") == (
        "patient00001_study1_view1_frontal"
    )


def test_frontal_filter(csv_path):
    assert len(read_chexpert_csv(csv_path)) == 3
    assert len(read_chexpert_csv(csv_path, frontal_only=False)) == 4


def test_uncertain_ignore_and_blank_zeros(csv_path):
    df = read_chexpert_csv(csv_path)
    p1 = df.loc["patient00001_study1_view1_frontal"]
    assert p1["Cardiomegaly"] == 1.0
    assert np.isnan(p1["Edema"])
    assert p1["Pneumonia"] == 0.0
    assert df.loc["patient00001_study1_view1_frontal", "patient_id"] == "patient00001"


@pytest.mark.parametrize("policy, expected", [("zeros", 0.0), ("ones", 1.0)])
def test_uncertain_policies(csv_path, policy, expected):
    df = read_chexpert_csv(csv_path, uncertain=policy)
    assert df.loc["patient00001_study1_view1_frontal", "Edema"] == expected
    assert not df[list(CHEXPERT_LABELS)].isna().any().any()


def test_blank_ignore(csv_path):
    df = read_chexpert_csv(csv_path, blank="ignore")
    assert np.isnan(df.loc["patient00002_study1_view1_frontal", "Pneumonia"])


def test_rejects_unknown_policy_and_values(csv_path, tmp_path):
    with pytest.raises(ValueError):
        read_chexpert_csv(csv_path, uncertain="maybe")
    bad = pd.read_csv(csv_path)
    bad.loc[0, "Cardiomegaly"] = 2.0
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="Unexpected label"):
        read_chexpert_csv(tmp_path / "bad.csv")


def test_split_has_no_patient_overlap():
    patients = pd.Series([f"p{i // 3}" for i in range(300)])
    split = split_by_patient(patients, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=0)
    assert split.notna().all()
    assert (pd.DataFrame({"p": patients, "s": split}).groupby("p")["s"].nunique() == 1).all()
    counts = split.groupby(patients).first().value_counts()
    assert counts.to_dict() == {"train": 80, "val": 10, "test": 10}


def test_split_is_seeded():
    patients = pd.Series([f"p{i}" for i in range(50)])
    frac = {"train": 0.5, "test": 0.5}
    a, b = split_by_patient(patients, frac, 0), split_by_patient(patients, frac, 0)
    assert a.equals(b)
    assert not a.equals(split_by_patient(patients, frac, 1))


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        split_by_patient(pd.Series(["a", "b"]), {"train": 0.5, "test": 0.4}, 0)


def test_load_gray_resizes_and_batch_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    src = tmp_path / "x.jpg"
    Image.fromarray(rng.integers(0, 256, (390, 320), dtype=np.uint8)).save(src)
    img = load_gray(src, 224)
    assert img.shape == (224, 224) and img.dtype == np.uint8

    save_png(img, tmp_path / "png" / "a.png")
    labels = pd.DataFrame({"Edema": [1.0]}, index=pd.Index(["a"], name="image_id"))
    images, y = load_batch(tmp_path / "png", labels, ["Edema"])
    assert images.shape == (1, 224, 224)
    np.testing.assert_array_equal(images[0], img)
    assert y.tolist() == [[1.0]]
