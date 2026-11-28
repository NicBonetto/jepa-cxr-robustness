import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from jepa_cxr.compare import SUMMARY_KEYS, make_fake_splits, run_comparison
from jepa_cxr.embeddings import save_embeddings
from jepa_cxr.extract import record
from jepa_cxr.sweep import (
    Condition,
    find_files,
    fixed_threshold_referral,
    load_split,
    plot_sweep,
    referral_threshold,
    run_sweep,
)
from jepa_cxr.uncertainty import KNNMethod, SoftmaxMethod

# find_files / load_split


def _write(root, split, corruption="clean", severity=0, encoder="radjepa", n=5, **over):
    name = "clean" if corruption == "clean" else f"{corruption}_s{severity}"
    path = root / encoder / "chexpert" / split / f"{name}.parquet"
    meta = dict(encoder=encoder, revision="r1", dataset="chexpert", split=split,
                corruption=corruption, severity=severity, seed=0) | over  # fmt: skip
    emb = np.zeros((n, 3), dtype=np.float32)
    save_embeddings(path, [f"{split}{i}" for i in range(n)], emb, **meta)
    record(root / "manifest.csv", path, emb, **meta)
    return path


@pytest.fixture
def grid(tmp_path):
    for split in ("train", "val", "test"):
        _write(tmp_path, split)
    for c in ("jpeg", "contrast"):
        for s in (2, 1):
            _write(tmp_path, "test", c, s)
    _write(tmp_path, "test", encoder="rad_dino")  # other encoder, must be ignored
    return tmp_path


def test_find_files(grid):
    train, val, conds = find_files(
        grid / "manifest.csv", "radjepa", "chexpert", "train", "val", "test"
    )
    assert train == grid / "radjepa/chexpert/train/clean.parquet"
    assert val == grid / "radjepa/chexpert/val/clean.parquet"
    assert [c.name for c in conds] == ["clean", "contrast_s1", "contrast_s2", "jpeg_s1", "jpeg_s2"]
    assert all(c.path.exists() for c in conds)


def test_find_files_needs_clean_splits(tmp_path):
    _write(tmp_path, "train")
    _write(tmp_path, "test")
    with pytest.raises(ValueError, match="clean radjepa/chexpert/val"):
        find_files(tmp_path / "manifest.csv", "radjepa", "chexpert", "train", "val", "test")
    with pytest.raises(ValueError, match="No vit"):
        find_files(tmp_path / "manifest.csv", "vit", "chexpert", "train", "val", "test")


@pytest.mark.parametrize("col,value", [("seed", 1), ("revision", "r2")])
def test_find_files_rejects_mixed_settings(grid, col, value):
    _write(grid, "test", "jpeg", 3, **{col: value})
    with pytest.raises(ValueError, match=f"mix {col}"):
        find_files(grid / "manifest.csv", "radjepa", "chexpert", "train", "val", "test")


def test_load_split_follows_file_order(tmp_path):
    path = _write(tmp_path, "test", n=3)
    labels = pd.DataFrame({"A": [1.0, 0.0, np.nan, 1.0]}, index=["test2", "test0", "test1", "x"])
    X, L = load_split(path, labels)
    assert list(L.index) == ["test0", "test1", "test2"]
    assert len(X) == 3
    with pytest.raises(ValueError, match="no labels"):
        load_split(path, labels.drop("test1"))


# referral


def test_referral_threshold_refers_requested_rate():
    u = np.random.default_rng(0).random(10_000)
    for rate in (0.1, 0.2, 0.5):
        assert np.mean(u > referral_threshold(u, rate)) == pytest.approx(rate, abs=0.01)
    with pytest.raises(ValueError, match="rate"):
        referral_threshold(u, 1.0)


def test_fixed_threshold_referral_values():
    errors = np.array([1.0, 0.0, 0.5, np.nan, 0.0])
    unc = np.array([0.9, 0.1, 0.6, 0.99, 0.2])
    referred, kept_err = fixed_threshold_referral(errors, unc, 0.5)
    assert referred == pytest.approx(2 / 4)  # NaN-error image left out
    assert kept_err == pytest.approx(0.0)
    referred, kept_err = fixed_threshold_referral(errors, unc, 0.0)
    assert referred == 1.0 and np.isnan(kept_err)


# run_sweep


@pytest.fixture(scope="module")
def splits():
    return make_fake_splits(n_train=600, n_val=300, n_test=400, dim=16, n_labels=3)


def _conditions(tests):
    conds = []
    for name in tests:
        corr, sev = ("clean", 0) if name == "clean" else ("shift", int(float(name[6:]) * 2))
        conds.append(Condition(corr, sev, name))  # path holds the key into `tests`
    return conds


def test_run_sweep_matches_run_comparison(splits):
    train, val, tests = splits
    t = run_sweep([SoftmaxMethod()], train, val, _conditions(tests), lambda c: tests[c.path])
    ref, _ = run_comparison([SoftmaxMethod()], train, val, tests)
    assert len(t) == len(tests)
    assert list(t["corruption"]) == ["clean", "shift", "shift", "shift"]
    assert list(t["severity"]) == [0, 1, 2, 4]
    np.testing.assert_allclose(t[list(SUMMARY_KEYS)], ref[list(SUMMARY_KEYS)])


def test_run_sweep_referral_columns(splits):
    train, val, tests = splits
    t = run_sweep(
        # Euclidean only: on isotropic fake data, cosine kNN distance cannot see additive noise,
        # and clean-val tuning picks cosine.
        [SoftmaxMethod(), KNNMethod(metrics=("euclidean",))], train, val, _conditions(tests),
        lambda c: tests[c.path], referral_rates=(0.1, 0.2),
    )  # fmt: skip
    assert {"referred@0.1", "kept_error@0.1", "referred@0.2", "kept_error@0.2"} <= set(t.columns)
    clean = t[t["corruption"] == "clean"]
    # Thresholds come from clean val, so clean test refers close to the requested rate.
    assert np.allclose(clean["referred@0.2"], 0.2, atol=0.07)
    # kNN distance grows with input shift, so the fixed threshold refers more and more cases.
    knn = t[t["method"] == "knn"].sort_values("severity")
    assert knn["referred@0.2"].is_monotonic_increasing
    assert knn["referred@0.2"].iloc[-1] > 0.5


def test_run_sweep_loads_each_condition_once(splits):
    train, val, tests = splits
    loaded = []

    def load(c):
        loaded.append(c.name)
        return tests[c.path]

    run_sweep([SoftmaxMethod(), KNNMethod()], train, val, _conditions(tests), load)
    assert sorted(loaded) == sorted(c.name for c in _conditions(tests))


def test_plot_sweep(splits):
    train, val, tests = splits
    t = run_sweep([SoftmaxMethod()], train, val, _conditions(tests), lambda c: tests[c.path])
    t2 = t.assign(corruption=t["corruption"].replace("shift", "other"))
    fig = plot_sweep(pd.concat([t, t2[t2["corruption"] == "other"]]), title="x")
    assert len(fig.axes) == 4 * 2  # 4 metrics x 2 corruptions
