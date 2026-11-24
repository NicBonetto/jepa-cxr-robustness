import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from jepa_cxr.compare import make_fake_splits, plot_referral_grid, run_comparison
from jepa_cxr.uncertainty import METHODS, UncertaintyMethod


@pytest.fixture(scope="module")
def splits():
    return make_fake_splits(n_train=600, n_val=200, n_test=300, dim=16, n_labels=3)


@pytest.mark.parametrize("name", list(METHODS))
def test_method_interface(name, splits):
    train, val, tests = splits
    method = METHODS[name]()
    assert isinstance(method, UncertaintyMethod)
    assert method.fit(*train, *val) is method

    X, Y = tests["clean"]
    probs, unc = method.predict(X)
    assert probs.shape == Y.shape
    assert unc.shape == (len(X),)
    assert np.isfinite(probs).all() and np.isfinite(unc).all()
    assert ((probs >= 0) & (probs <= 1)).all()


@pytest.mark.parametrize("name", list(METHODS))
def test_method_deterministic(name, splits):
    train, val, tests = splits
    X = tests["clean"][0]
    a = METHODS[name]().fit(*train, *val).predict(X)
    b = METHODS[name]().fit(*train, *val).predict(X)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_fake_splits_structure():
    train, _val, tests = make_fake_splits(n_train=100, n_val=50, n_test=40, dim=8, n_labels=2)
    assert np.isnan(train[1]).any()
    assert not np.isnan(tests["clean"][1]).any()
    assert set(tests) == {"clean", "shift_0.5", "shift_1", "shift_2"}
    np.testing.assert_array_equal(tests["clean"][1], tests["shift_2"][1])


def test_run_comparison_and_plot(splits):
    train, val, tests = splits
    methods = [cls() for cls in METHODS.values()]
    results, outputs = run_comparison(methods, train, val, tests)
    assert len(results) == len(methods) * len(tests)
    assert {"method", "condition", "macro_auroc", "aurc"} <= set(results.columns)

    # Shifting embeddings should hurt the classifier
    auroc = results.pivot(index="method", columns="condition", values="macro_auroc")
    assert (auroc["shift_2"] < auroc["clean"]).all()

    plot_referral_grid(outputs, tests)

