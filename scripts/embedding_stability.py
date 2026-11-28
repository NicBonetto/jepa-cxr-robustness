"""Aim 1: stability table and severity plot for every corrupted condition in the manifest.

Example:
    uv run python scripts/embedding_stability.py
    uv run python scripts/embedding_stability.py --manifest /tmp/emb/manifest.csv --out-dir /tmp/s
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import yaml

from jepa_cxr.stability import plot_severity, stability_table


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/stability.yaml"))
    p.add_argument("--manifest", type=Path, default=None, help="Default: from --config")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: from --config")
    p.add_argument("--k", type=int, default=None, help="Default: from --config")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    manifest = args.manifest or Path(cfg["manifest"])
    out_dir = args.out_dir or Path(cfg["out_dir"])
    k = args.k or cfg["k"]

    table = stability_table(manifest, k, cfg.get("encoders"), cfg.get("datasets"))
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "stability.csv", index=False)
    plot_severity(table).savefig(out_dir / "severity.png", dpi=150)
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg | {"manifest": str(manifest), "out_dir": str(out_dir), "k": k})
    )
    print(table.round(3).to_string(index=False))
    print(f"\nSaved stability.csv, severity.png and config.yaml to {out_dir}")


if __name__ == "__main__":
    main()
