"""Grayscale image loading and PNG caching shared by the dataset loaders."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def resize(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize a uint8 grayscale array to (size, size)."""
    if img.shape == (size, size):
        return img
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def load_gray(path: str | Path, size: int = 224) -> np.ndarray:
    """Load an image as uint8 grayscale, resized to (size, size)."""
    with Image.open(path) as im:
        return resize(np.asarray(im.convert("L")), size)


def save_png(img: np.ndarray, path: str | Path) -> None:
    """Write a uint8 grayscale array as PNG, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def load_batch(
    png_dir: str | Path, labels: pd.DataFrame, label_cols, n: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Cached PNGs (N, H, W) uint8 and labels (N, L), in `labels` row order."""
    rows = labels.iloc[:n] if n is not None else labels
    images = np.stack([load_gray(Path(png_dir) / f"{i}.png") for i in rows.index])
    return images, rows[list(label_cols)].to_numpy(dtype=float)
