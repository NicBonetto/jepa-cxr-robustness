import numpy as np
import pandas as pd
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from jepa_cxr.images import resize
from jepa_cxr.vindr import read_dicom, read_vindr_labels


def _write_dicom(path, arr, photometric="MONOCHROME2", window=None):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.set_pixel_data(arr, photometric, 12)
    if window:
        ds.WindowCenter, ds.WindowWidth = window
    ds.save_as(path, enforce_file_format=True)
    return path


@pytest.fixture
def ramp():
    return np.tile(np.linspace(0, 4095, 64).astype(np.uint16), (48, 1))


def test_majority_and_any(tmp_path):
    rows = [
        {"image_id": "a", "rad_id": "R1", "Nodule/Mass": 1, "Pneumonia": 1},
        {"image_id": "a", "rad_id": "R2", "Nodule/Mass": 1, "Pneumonia": 0},
        {"image_id": "a", "rad_id": "R3", "Nodule/Mass": 0, "Pneumonia": 0},
        {"image_id": "b", "rad_id": "R1", "Nodule/Mass": 0, "Pneumonia": 0},
    ]
    path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    majority = read_vindr_labels(path)
    assert list(majority.columns) == ["Nodule/Mass", "Pneumonia"]
    assert majority.loc["a"].tolist() == [1.0, 0.0]
    assert majority.loc["b"].tolist() == [0.0, 0.0]
    assert read_vindr_labels(path, "any").loc["a"].tolist() == [1.0, 1.0]


def test_rejects_bad_values_and_policy(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame([{"image_id": "a", "Pneumonia": 2}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unexpected label"):
        read_vindr_labels(path)
    with pytest.raises(ValueError):
        read_vindr_labels(path, "vote")


def test_read_dicom_full_range(tmp_path, ramp):
    img = read_dicom(_write_dicom(tmp_path / "x.dicom", ramp))
    assert img.shape == ramp.shape and img.dtype == np.uint8
    assert img.min() == 0 and img.max() == 255
    assert (np.diff(img[0].astype(int)) >= 0).all()


def test_monochrome1_is_inverted(tmp_path, ramp):
    m2 = read_dicom(_write_dicom(tmp_path / "m2.dicom", ramp))
    m1 = read_dicom(_write_dicom(tmp_path / "m1.dicom", ramp, "MONOCHROME1"))
    np.testing.assert_array_equal(m1, 255 - m2)


def test_window_is_applied(tmp_path, ramp):
    img = read_dicom(_write_dicom(tmp_path / "w.dicom", ramp, window=(2048, 1024)))
    assert (img[0, :10] == 0).all() and (img[0, -10:] == 255).all()


def test_constant_image_does_not_divide_by_zero(tmp_path):
    img = read_dicom(_write_dicom(tmp_path / "c.dicom", np.full((8, 8), 100, np.uint16)))
    assert (img == 0).all()


def test_resize_to_224(tmp_path, ramp):
    assert resize(read_dicom(_write_dicom(tmp_path / "x.dicom", ramp))).shape == (224, 224)
