"""Embed a folder of chest X-ray PNGs, optionally corrupted.

Example:
    uv run python scripts/extract_embeddings.py --encoder radjepa --images data/samples \
        --out embeddings/radjepa/samples/gaussian_noise_s3.parquet \
        --corruption gaussian_noise --severity 3
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from jepa_cxr.corruptions import corrupt
from jepa_cxr.embeddings import embed, load_encoder, load_encoder_config, save_embeddings

CORRUPT_SIZE = 224

def load_png(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").resize((CORRUPT_SIZE, CORRUPT_SIZE), Image.BILINEAR)
    return np.asarray(img)

def resize(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size:
        return img
    out = Image.fromarray(img, mode="F").resize((size, size), Image.BICUBIC)
    return np.clip(np.asarray(out), 0, 1).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True, help="Key in configs/encoders.yaml")
    p.add_argument("--config", type=Path, default=Path("configs/encoders.yaml"))
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--corruption", default=None)
    p.add_argument("--severity", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="Embed only the first N images")
    args = p.parse_args()

    cfg = load_encoder_config(args.encoder, args.config)
    paths = sorted(args.images.glob("*.png"))[: args.limit]

    images = []
    for i, path in enumerate(paths):
        img = load_png(path)
        if args.corruption and args.severity > 0:
            img = corrupt(img, args.corruption, args.severity, seed=args.seed + i)
        else:
            img = img.astype(np.float32) / 255
        images.append(resize(img, cfg["image_size"]))

    model = load_encoder(cfg["hf_name"], cfg["revision"], cfg.get("trust_remote_code", False))
    emb = embed(model, images, cfg["mean"], cfg["std"], cfg["pooling"], args.batch_size)

    save_embeddings(
        args.out,
        [p.stem for p in paths],
        emb,
        encoder=args.encoder,
        revision=cfg["revision"] or "unpinned",
        corruption=args.corruption or "clean",
        severity=args.severity,
        seed=args.seed,
    )
    print(f"Saved {emb.shape} to {args.out}")


if __name__ == "__main__":
    main()
