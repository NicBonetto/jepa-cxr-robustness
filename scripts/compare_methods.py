import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from jepa_cxr.compare import make_fake_splits, plot_referral_grid, run_comparison
from jepa_cxr.embeddings import load_embeddings
from jepa_cxr.labels import frequent_labels
from jepa_cxr.uncertainty import METHODS


def load_split(emb_path: str, labels: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    ids, X = load_embeddings(emb_path)
    return X, labels.loc[ids]


def read_labels(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(path).set_index("image_id") for path in paths]
    if any(list(f.columns) != list(frames[0].columns) for f in frames):
        raise ValueError("Label files must share columns (use labels_shared.csv) across datasets")
    return pd.concat(frames, verify_integrity=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--fake", action="store_true", help="Use synthetic embeddings")
    p.add_argument("--train")
    p.add_argument("--val")
    p.add_argument("--test", action="append", default=[], help="name=path, repeatable")
    p.add_argument("--labels", action="append", default=[], help="labels CSV, repeatable")
    p.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    p.add_argument("--min-positives", type=int, default=None, help="Default: from --config")
    args = p.parse_args()

    if args.fake:
        train, val, tests = make_fake_splits()
    else:
        if not (args.train and args.val and args.test and args.labels):
            p.error("real data needs --train, --val, at least one --test, and --labels")
        labels = read_labels(args.labels)
        train, val = load_split(args.train, labels), load_split(args.val, labels)
        tests = {}
        for spec in args.test:
            name, path = spec.split("=", 1)
            tests[name] = load_split(path, labels)

        min_pos = args.min_positives
        if min_pos is None:
            min_pos = yaml.safe_load(args.config.read_text())["labels"]["min_positives"]
        keep = frequent_labels([Y for _, Y in tests.values()], min_pos)
        print(
            f"Labels with < {min_pos} positives in a test set, dropped: "
            f"{[c for c in labels.columns if c not in keep] or 'none'}"
        )

        def select(split):
            return split[0], split[1][keep].to_numpy(dtype=float)

        train, val = select(train), select(val)
        tests = {name: select(split) for name, split in tests.items()}

    methods = [METHODS[name]() for name in args.methods]
    results, outputs = run_comparison(methods, train, val, tests)

    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "results.csv", index=False)
    plot_referral_grid(outputs, tests).savefig(args.out / "referral.png", dpi=150)
    print(results.round(3).to_string(index=False))
    print(f"\nSaved results.csv and referral.png to {args.out}")


if __name__ == "__main__":
    main()
