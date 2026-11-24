"""Convert VinDr-CXR DICOMs to PNGs and write label, split and config files (see docs/data.md)."""

import argparse
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from jepa_cxr.images import resize, save_png
from jepa_cxr.labels import VINDR_MAP, to_shared
from jepa_cxr.vindr import read_dicom, read_vindr_labels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root", type=Path, required=True, help="Folder with train/, test/, annotations/"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    p.add_argument(
        "--limit", type=int, default=None, help="Process only the first N images per split"
    )
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["vindr"]
    frames = []
    for split in cfg["splits"]:
        labels = read_vindr_labels(
            args.root / "annotations" / f"image_labels_{split}.csv", cfg["aggregate"]
        ).iloc[: args.limit]
        for image_id in tqdm(labels.index, desc=split):
            out = args.out / "images" / split / f"{image_id}.png"
            if not out.exists():
                dicom = args.root / split / f"{image_id}{cfg['dicom_ext']}"
                save_png(resize(read_dicom(dicom), cfg["image_size"]), out)
        frames.append(labels.assign(split=split))

    df = pd.concat(frames)
    df.drop(columns="split").to_csv(args.out / "labels.csv")
    shared = to_shared(df, VINDR_MAP)
    shared.to_csv(args.out / "labels_shared.csv")
    df[["split"]].to_csv(args.out / "splits.csv")
    (args.out / "config.yaml").write_text(yaml.safe_dump({"vindr": cfg}))
    print(shared.groupby(df["split"]).sum().T.astype(int).to_string())
    print(f"Saved {len(df)} images to {args.out}")


if __name__ == "__main__":
    main()
