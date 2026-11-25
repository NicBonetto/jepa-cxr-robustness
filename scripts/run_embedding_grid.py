"""Embed every condition in configs/embeddings.yaml, skipping ones already in the manifest.

Example:
    uv run python scripts/run_embedding_grid.py --dry-run
    uv run python scripts/run_embedding_grid.py --encoder radjepa --dataset vindr
"""

import argparse
from itertools import groupby
from pathlib import Path

import yaml

from jepa_cxr.embeddings import (
    DEVICES,
    get_device,
    load_encoder,
    load_encoder_config,
    save_embeddings,
)
from jepa_cxr.extract import build_grid, embed_paths, is_done, read_manifest, record


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=DEVICES, default="auto", help="Default: auto-detect")
    p.add_argument("--config", type=Path, default=Path("configs/embeddings.yaml"))
    p.add_argument("--encoders-config", type=Path, default=Path("configs/encoders.yaml"))
    p.add_argument("--encoder", action="append", help="Only this encoder, repeatable")
    p.add_argument("--dataset", action="append", help="Only this dataset, repeatable")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: out_dir from --config")
    p.add_argument("--limit", type=int, default=None, help="First N images per split")
    p.add_argument("--force", action="store_true", help="Redo jobs already in the manifest")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = args.out_dir or Path(cfg["out_dir"])
    if args.limit is not None and args.out_dir is None:
        p.error("--limit needs --out-dir, so partial runs never land in the real manifest")
    manifest = out_dir / "manifest.csv"

    jobs = [
        j
        for j in build_grid(cfg)
        if (not args.encoder or j.encoder in args.encoder)
        and (not args.dataset or j.dataset in args.dataset)
    ]
    done = read_manifest(manifest)
    todo = [j for j in jobs if args.force or not is_done(done, j.path(out_dir), manifest)]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    if args.dry_run:
        for j in todo:
            print(j.path(out_dir))
        return

    device = get_device(args.device)
    print(f"Device: {device}")

    for enc, enc_jobs in groupby(todo, key=lambda j: j.encoder):
        enc_cfg = load_encoder_config(enc, args.encoders_config)
        model = load_encoder(
            enc_cfg["hf_name"], enc_cfg["revision"], enc_cfg["trust_remote_code"], device
        )
        for j in enc_jobs:
            src = Path(cfg["datasets"][j.dataset]["images"]) / j.split
            paths = sorted(src.glob("*.png"))[: args.limit]
            if not paths:
                raise FileNotFoundError(f"No PNGs in {src}")
            emb = embed_paths(
                model, paths, enc_cfg, j.corruption, j.severity, cfg["seed"],
                cfg["batch_size"], cfg["chunk_size"],
            )  # fmt: skip
            meta = dict(
                encoder=enc, revision=enc_cfg["revision"], dataset=j.dataset, split=j.split,
                corruption=j.corruption, severity=j.severity, seed=cfg["seed"],
            )  # fmt: skip
            out = j.path(out_dir)
            save_embeddings(out, [p.stem for p in paths], emb, **meta)
            record(manifest, out, emb, **meta)
            print(f"Saved {emb.shape} to {out}")


if __name__ == "__main__":
    main()
