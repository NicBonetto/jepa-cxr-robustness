"""Embedding job grid, image preparation and the embedding manifest."""

import hashlib
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from jepa_cxr.corruptions import CORRUPTIONS, corrupt
from jepa_cxr.embeddings import embed

CORRUPT_SIZE = 224
MANIFEST_COLUMNS = [
    "path", "encoder", "revision", "dataset", "split", "corruption", "severity", "seed",
    "n_images", "dim", "sha256", "git_commit", "created_at",
]  # fmt: skip


@dataclass(frozen=True)
class Job:
    encoder: str
    dataset: str
    split: str
    corruption: str
    severity: int

    def path(self, out_dir: str | Path) -> Path:
        """Output parquet path under `out_dir`."""
        name = "clean" if self.corruption == "clean" else f"{self.corruption}_s{self.severity}"
        return Path(out_dir) / self.encoder / self.dataset / self.split / f"{name}.parquet"


def build_grid(cfg: dict) -> list[Job]:
    """All jobs in an embeddings config: clean splits, then corrupted splits."""
    unknown = set(cfg["corruptions"]) - set(CORRUPTIONS)
    if unknown:
        raise ValueError(f"Unknown corruptions {sorted(unknown)}; choose from {CORRUPTIONS}")
    if not all(1 <= s <= 5 for s in cfg["severities"]):
        raise ValueError(f"Severities must be 1-5, got {cfg['severities']}")

    jobs = []
    for enc in cfg["encoders"]:
        for ds, d in cfg["datasets"].items():
            jobs += [Job(enc, ds, split, "clean", 0) for split in d.get("clean", [])]
            for split in d.get("corrupt", []):
                for c in cfg["corruptions"]:
                    jobs += [Job(enc, ds, split, c, s) for s in cfg["severities"]]
    if len(set(jobs)) != len(jobs):
        raise ValueError("Embeddings config produces duplicate jobs")
    return jobs


def load_png(path: str | Path) -> np.ndarray:
    """Load a PNG as uint8 grayscale at the corruption size."""
    with Image.open(path) as im:
        img = im.convert("L").resize((CORRUPT_SIZE, CORRUPT_SIZE), Image.BILINEAR)
        return np.asarray(img)


def prepare(img: np.ndarray, corruption: str, severity: int, seed: int, size: int) -> np.ndarray:
    """Corrupt a uint8 image at 224x224, then resize to the encoder size as float32 in [0, 1]."""
    if corruption == "clean":
        x = img.astype(np.float32) / 255
    else:
        x = corrupt(img, corruption, severity, seed=seed)
    if x.shape[0] == size:
        return x
    out = Image.fromarray(x, mode="F").resize((size, size), Image.BICUBIC)
    return np.clip(np.asarray(out), 0, 1).astype(np.float32)


def embed_paths(
    model,
    paths: list[Path],
    enc_cfg: dict,
    corruption: str = "clean",
    severity: int = 0,
    seed: int = 0,
    batch_size: int = 64,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Embed PNGs in chunks. Image i gets corruption seed (seed + i)."""
    out = []
    for start in range(0, len(paths), chunk_size):
        images = [
            prepare(load_png(p), corruption, severity, seed + start + j, enc_cfg["image_size"])
            for j, p in enumerate(paths[start : start + chunk_size])
        ]
        out.append(
            embed(model, images, enc_cfg["mean"], enc_cfg["std"], enc_cfg["pooling"], batch_size)
        )
    return np.concatenate(out)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    """HEAD commit, suffixed -dirty if tracked files changed, or 'unknown'."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"]).returncode != 0
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


def read_manifest(manifest: str | Path) -> pd.DataFrame:
    if not Path(manifest).exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return pd.read_csv(manifest, keep_default_na=False)


def _rel(path: str | Path, manifest: str | Path) -> str:
    return os.path.relpath(path, Path(manifest).parent)


def record(manifest: str | Path, emb_path: str | Path, emb: np.ndarray, **meta) -> None:
    """Add or replace the manifest row for `emb_path`. Paths are relative to the manifest."""
    row = {
        "path": _rel(emb_path, manifest),
        "n_images": emb.shape[0],
        "dim": emb.shape[1],
        "sha256": file_sha256(emb_path),
        "git_commit": git_commit(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **meta,
    }
    missing = set(MANIFEST_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"Manifest row is missing {sorted(missing)}")
    df = read_manifest(manifest)
    df = pd.concat([df[df["path"] != row["path"]], pd.DataFrame([row])], ignore_index=True)
    Path(manifest).parent.mkdir(parents=True, exist_ok=True)
    df[MANIFEST_COLUMNS].to_csv(manifest, index=False)


def is_done(manifest_df: pd.DataFrame, emb_path: str | Path, manifest: str | Path) -> bool:
    """True if `emb_path` exists and has a manifest row."""
    return Path(emb_path).exists() and _rel(emb_path, manifest) in set(manifest_df["path"])
