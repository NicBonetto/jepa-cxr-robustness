"""Convert CheXpert to PNGs and write label, split and config files (see docs/data.md)."""

import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

from jepa_cxr.chexpert import CHEXPERT_LABELS, read_chexpert_csv, split_by_patient
from jepa_cxr.images import load_gray, save_png
from jepa_cxr.labels import CHEXPERT_MAP, to_shared


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    p.add_argument("--limit", type=int, default=None, help="Process only the first N images")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["chexpert"]
    df = read_chexpert_csv(args.csv, cfg["uncertain"], cfg["blank"], cfg["frontal_only"])
    df["split"] = split_by_patient(df["patient_id"], cfg["split"], cfg["seed"])
    df = df.iloc[: args.limit]

    for image_id, row in tqdm(df.iterrows(), total=len(df)):
        out = args.out / "images" / row["split"] / f"{image_id}.png"
        if not out.exists():
            save_png(load_gray(args.root / row["path"], cfg["image_size"]), out)

    df[list(CHEXPERT_LABELS)].to_csv(args.out / "labels.csv")
    to_shared(df, CHEXPERT_MAP).to_csv(args.out / "labels_shared.csv")
    df[["patient_id", "split"]].to_csv(args.out / "splits.csv")
    (args.out / "config.yaml").write_text(yaml.safe_dump({"chexpert": cfg, "csv": str(args.csv)}))
    print(df["split"].value_counts().to_string())
    print(f"Saved {len(df)} images to {args.out}")


if __name__ == "__main__":
    main()
