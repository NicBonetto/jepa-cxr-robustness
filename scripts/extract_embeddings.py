"""Embed one folder of chest X-ray PNGs, optionally corrupted. For the full grid, use
scripts/run_embedding_grid.py.

Example:
    uv run python scripts/extract_embeddings.py --encoder radjepa --images data/samples \
        --out embeddings/radjepa/samples/gaussian_noise_s3.parquet \
        --corruption gaussian_noise --severity 3
"""

import argparse
from pathlib import Path

from jepa_cxr.embeddings import (
    DEVICES,
    get_device,
    load_encoder,
    load_encoder_config,
    save_embeddings,
)
from jepa_cxr.extract import embed_paths, record


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True, help="Key in configs/encoders.yaml")
    p.add_argument("--device", choices=DEVICES, default="auto", help="Default: auto-detect")
    p.add_argument("--config", type=Path, default=Path("configs/encoders.yaml"))
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("embeddings/manifest.csv"))
    p.add_argument("--corruption", default="clean")
    p.add_argument("--severity", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="Embed only the first N images")
    args = p.parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    corruption = args.corruption if args.severity > 0 else "clean"
    severity = args.severity if corruption != "clean" else 0
    cfg = load_encoder_config(args.encoder, args.config)
    paths = sorted(args.images.glob("*.png"))[: args.limit]

    model = load_encoder(
        cfg["hf_name"], cfg["revision"], cfg.get("trust_remote_code", False), device
    )
    emb = embed_paths(model, paths, cfg, corruption, severity, args.seed, args.batch_size)

    meta = dict(
        encoder=args.encoder, revision=cfg["revision"] or "unpinned", dataset=args.images.name,
        split="", corruption=corruption, severity=severity, seed=args.seed,
    )  # fmt: skip
    save_embeddings(args.out, [p.stem for p in paths], emb, **meta)
    record(args.manifest, args.out, emb, **meta)
    print(f"Saved {emb.shape} to {args.out}")


if __name__ == "__main__":
    main()
