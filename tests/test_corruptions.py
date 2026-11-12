import numpy as np
import pytest
 
from jepa_cxr.corruptions import CORRUPTIONS, corrupt
 
SEVERITIES = (1, 2, 3, 4, 5)
 
 
@pytest.fixture
def img() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
 
 
@pytest.mark.parametrize("name", CORRUPTIONS)
@pytest.mark.parametrize("severity", SEVERITIES)
def test_shape_dtype_range(img, name, severity):
    out = corrupt(img, name, severity, seed=1)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
 
 
@pytest.mark.parametrize("name", CORRUPTIONS)
def test_severity_zero_is_identity(img, name):
    np.testing.assert_allclose(corrupt(img, name, 0), img / 255.0, atol=1e-6)
 
 
@pytest.mark.parametrize("name", CORRUPTIONS)
def test_same_seed_is_deterministic(img, name):
    a = corrupt(img, name, 3, seed=42)
    b = corrupt(img, name, 3, seed=42)
    np.testing.assert_array_equal(a, b)
 
 
@pytest.mark.parametrize("name", CORRUPTIONS)
def test_damage_increases_with_severity(img, name):
    clean = img / 255.0
    errs = [np.abs(corrupt(img, name, s, seed=7) - clean).mean() for s in SEVERITIES]
    assert errs[-1] > errs[0]

def test_poisson_noise_stronger_in_dense_regions():
    img = np.full((64, 64), 0.2, dtype=np.float32)
    img[:, 32:] = 0.8  # right half = dense tissue
    out = corrupt(img, "poisson", 3, seed=0)
    assert out[:, 32:].std() > out[:, :32].std()
 
def test_invalid_inputs(img):
    with pytest.raises(ValueError):
        corrupt(img, "not_a_corruption", 1)
    with pytest.raises(ValueError):
        corrupt(img, "gaussian_noise", 6)
