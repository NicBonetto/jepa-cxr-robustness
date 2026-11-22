import numpy as np

from jepa_cxr.classifiers import fit_logistic, fit_temperature, nll, select_l2
from jepa_cxr.metrics import ece, performance


def make_data(n=600, d=20, n_labels=3, noise=1.0, seed=0):
    """Fake embeddings whose labels depend linearly on the features, plus noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    true_w = rng.normal(size=(d, n_labels))
    logits = X @ true_w + rng.normal(scale=noise, size=(n, n_labels))
    return X, (logits > 0).astype(float)


def test_learns_separable_signal():
    X, Y = make_data(noise=0.1)
    head = fit_logistic(X[:400], Y[:400])
    p = head.predict_proba(X[400:])
    assert p.shape == (200, 3)
    assert performance(Y[400:], p)["macro_auroc"] > 0.95


def test_missing_labels_are_ignored():
    X, Y = make_data()
    Y_missing = Y.copy()
    Y_missing[::3, 0] = np.nan  # a third of label 0 unknown
    head = fit_logistic(X, Y_missing)
    assert np.isfinite(head.W).all()
    keep = ~np.isnan(Y_missing[:, 0])
    ref = fit_logistic(X[keep], Y[keep, :1])
    # Label 0 should match a fit on only its labeled rows (up to standardization differences)
    corr = np.corrcoef(head.logits(X)[:, 0], ref.logits(X)[:, 0])[0, 1]
    assert corr > 0.99


def test_temperature_fixes_overconfidence_without_changing_ranking():
    X, Y = make_data(n=1200, d=60, noise=3.0, seed=1)
    tr, va, te = slice(0, 300), slice(300, 800), slice(800, 1200)
    head = fit_logistic(X[tr], Y[tr], l2=1e-6)  # few samples, weak penalty -> overconfident
    scaled = fit_temperature(head, X[va], Y[va])

    assert scaled.temperature > 1.0
    assert ece(Y[te], scaled.predict_proba(X[te])) < ece(Y[te], head.predict_proba(X[te]))
    assert nll(scaled, X[te], Y[te]) < nll(head, X[te], Y[te])

    np.testing.assert_allclose(
        performance(Y[te], scaled.logits(X[te]) / scaled.temperature)["auroc"],
        performance(Y[te], head.logits(X[te]))["auroc"],
    )


def test_select_l2_returns_grid_value():
    X, Y = make_data()
    grid = (1e-4, 1e-2, 1.0)
    assert select_l2(X[:400], Y[:400], X[400:], Y[400:], grid) in grid


def test_single_label_input():
    X, Y = make_data(n_labels=1)
    head = fit_logistic(X, Y[:, 0])
    assert head.predict_proba(X).shape == (600, 1)


def test_deterministic():
    X, Y = make_data()
    a, b = fit_logistic(X, Y), fit_logistic(X, Y)
    np.testing.assert_array_equal(a.W, b.W)


def test_stronger_penalty_shrinks_weights():
    X, Y = make_data()
    weak, strong = fit_logistic(X, Y, 1e-4), fit_logistic(X, Y, 1.0)
    assert np.abs(strong.W).sum() < np.abs(weak.W).sum()


def test_label_with_one_class_predicts_its_rate():
    X, Y = make_data()
    Y[:, 1] = 0.0
    head = fit_logistic(X, Y)
    assert np.allclose(head.W[:, 1], 0)
    assert head.predict_proba(X)[:, 1].max() < 0.01
