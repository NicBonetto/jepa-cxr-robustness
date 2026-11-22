import argparse
from pathlib import Path

import pandas as pd

from jepa_cxr.compare import make_fake_splits, plot_referral_grid, run_comparison
from jepa_cxr.embeddings import load_embeddings
from jepa_cxr.uncertainty import METHODS


def load_split(emb_path: str, labels: pd.DataFrame):
    ids, X = load_embeddings(emb_path)
    Y = labels.loc[ids].to_numpy(dtype=float)  # rows aligned to embedding order
    return X, Y


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--fake", action="store_true", help="Use synthetic embeddings")
    p.add_argument("--train")
    p.add_argument("--val")
    p.add_argument("--test", action="append", default=[], help="name=path, repeatable")
    p.add_argument("--labels", help="CSV with image_id + one column per finding")
    args = p.parse_args()

    if args.fake:
        train, val, tests = make_fake_splits()
    else:
        if not (args.train and args.val and args.test and args.labels):
            p.error("real data needs --train, --val, at least one --test, and --labels")
        labels = pd.read_csv(args.labels).set_index("image_id")
        train, val = load_split(args.train, labels), load_split(args.val, labels)
        tests = {}
        for spec in args.test:
            name, path = spec.split("=", 1)
            tests[name] = load_split(path, labels)

    methods = [METHODS[name]() for name in args.methods]
    results, outputs = run_comparison(methods, train, val, tests)

    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "results.csv", index=False)
    plot_referral_grid(outputs, tests).savefig(args.out / "referral.png", dpi=150)
    print(results.round(3).to_string(index=False))
    print(f"\nSaved results.csv and referral.png to {args.out}")


if __name__ == "__main__":
    main()

