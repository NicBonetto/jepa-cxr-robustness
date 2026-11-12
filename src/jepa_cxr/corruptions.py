"""Chest X-ray corruptions.
 
Input: 2D grayscale image, uint8 or float in [0, 1]. Apply at a fixed size (224x224) before the
encoder's normalization. Output: float32 in [0, 1], same shape.

All images are assumed to be MONOCHROME2.
"""

import io

import numpy as np
from PIL import Image, ImageFilter

N0 = 20000

PARAMS = {
    "gaussian_noise": (0.02, 0.04, 0.06, 0.09, 0.12),
    "gaussian_blur": (0.5, 1.0, 1.5, 2.5, 3.5),
    "contrast": (0.80, 0.65, 0.50, 0.35, 0.25),
    "poisson": tuple(N0 / s**2 for s in (1, 2, 4, 6, 9)),
    "jpeg": (60, 40, 25, 15, 8)
}
CORRUPTIONS = tuple(PARAMS)

def _pil_op(x, op):
    out = op(Image.fromarray((x * 255).round().astype(np.uint8)))
    return np.asarray(out, dtype=np.float32) / 255

def _quantum_noise(x, photons, rng, k=3.0):
    counts = rng.poisson(photons * np.exp(-k * x))
    return -np.log(np.maximum(counts, 1) / photons) / k

def _jpeg(pil, quality):
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    return Image.open(buf)

FUNCS = {
    "gaussian_noise": lambda x, p, rng: x + rng.normal(0, p, x.shape),
    "gaussian_blur": lambda x, p, rng: _pil_op(
        x, lambda im: im.filter(ImageFilter.GaussianBlur(p))
    ),
    "contrast": lambda x, p, rng: x.mean() + (x - x.mean()) * p,
    "poisson": lambda x, p, rng: _quantum_noise(x, p, rng),
    "jpeg": lambda x, p, rng: _pil_op(x, lambda im: _jpeg(im, p)),
}

def corrupt(img: np.ndarray, corruption: str, severity: int, seed: int | None = None) -> np.ndarray:
    """Apply `corruption` at `severity` (0 = unchanged, 1-5)."""
    if corruption not in PARAMS:
        raise ValueError(f"Unknown corruption {corruption}; choose from {CORRUPTIONS}")
    if not 0 <= severity <= 5:
        raise ValueError(f"Severity must be 0-5, got {severity}")
 
    x = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img.astype(np.float32)
    if severity == 0:
        return x
 
    out = FUNCS[corruption](x, PARAMS[corruption][severity - 1], np.random.default_rng(seed))
    return np.clip(out, 0, 1).astype(np.float32)
