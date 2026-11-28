"""Aim 2: fit each method on clean CheXpert, evaluate on every corrupted test condition.

Writes results.csv (all encoders) and one sweep_<encoder>.png per encoder.

Example:
    uv run python scripts/corruption_sweep.py
    uv run python scripts/corruption_sweep.py --encoder radjepa --out-dir /tmp/aim2
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import yaml

from jepa_cxr.labels import frequent_labels
from jepa_cxr.sweep import find_files, load_split, plot_sweep, run_sweep
from jepa_cxr.uncertainty import METHODS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/aim2.yaml"))
    p.add_argument("--manifest", type=Path, default=None, help="Default: from --config")
    p.add_argument("--labels", type=Path, default=None, help="Default: from --config")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: from --config")
    p.add_argument("--encoder", action="append", help="Only this encoder, repeatable")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    cfg["manifest"] = str(args.manifest or cfg["manifest"])
    cfg["labels"] = str(args.labels or cfg["labels"])
    cfg["out_dir"] = str(args.out_dir or cfg["out_dir"])
    cfg["encoders"] = args.encoder or cfg["encoders"]
    unknown = set(cfg["methods"]) - set(METHODS)
    if unknown:
        p.error(f"Unknown methods {sorted(unknown)}; choose from {list(METHODS)}")
    min_pos = yaml.safe_load(Path(cfg["data_config"]).read_text())["labels"]["min_positives"]
    sp = cfg["splits"]
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(cfg["labels"]).set_index("image_id")
    tables = []
    for enc in cfg["encoders"]:
        train_p, val_p, conditions = find_files(
            cfg["manifest"], enc, cfg["dataset"], sp["train"], sp["val"], sp["test"]
        )
        X_tr, L_tr = load_split(train_p, labels)
        X_va, L_va = load_split(val_p, labels)
        # Labels are identical across conditions, so the clean test set decides which to keep.
        _, L_te = load_split(conditions[0].path, labels)
        keep = frequent_labels([L_te], min_pos)
        dropped = [c for c in labels.columns if c not in keep]
        print(f"[{enc}] {len(conditions)} test conditions; labels dropped (< {min_pos} "
              f"positives in test): {dropped or 'none'}")  # fmt: skip

        def as_array(L, keep=keep):
            return L[keep].to_numpy(dtype=float)

        def load_test(cond, keep=keep):
            X, L = load_split(cond.path, labels)
            return X, as_array(L, keep)

        methods = [METHODS[name]() for name in cfg["methods"]]
        t = run_sweep(
            methods, (X_tr, as_array(L_tr)), (X_va, as_array(L_va)), conditions, load_test,
            tuple(cfg["referral_rates"]),
        )  # fmt: skip
        t.insert(0, "encoder", enc)
        t["labels"] = ",".join(keep)
        tables.append(t)
        plot_sweep(t, title=f"{enc} · {cfg['dataset']}").savefig(
            out_dir / f"sweep_{enc}.png", dpi=150
        )

    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg | {"min_positives": min_pos}))
    show = ["encoder", "method", "corruption", "severity", "macro_auroc", "ece", "aurc",
            "loss_spearman", "referred@0.2"]  # fmt: skip
    print(table[show].round(3).to_string(index=False))
    print(f"\nSaved results.csv, sweep_*.png and config.yaml to {out_dir}")


if __name__ == "__main__":
    main()
