import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from jepa_cxr.metrics import (
    aurc,
    ece,
    entropy,
    evaluate,
    image_brier,
    image_uncertainty,
    knn_distance,
    loss_spearman,
    performance,
    plot_referral,
    plot_reliability,
    risk_coverage,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_perfect_predictions(rng):
    y = rng.integers(0, 2, size=(200, 3))
    res = performance(y, y.astype(float))
    assert res["macro_auroc"] == 1.0
    assert res["macro_auprc"] == 1.0
    assert ece(y, y.astype(float)) == 0.0


def test_single_class_label_is_nan(rng):
    y = rng.integers(0, 2, size=(50, 2))
    y[:, 1] = 0
    res = performance(y, rng.random((50, 2)))
    assert np.isnan(res["auroc"][1]) and not np.isnan(res["macro_auroc"])


def test_ece_known_values():
    y = np.array([1] * 80 + [0] * 20)
    assert ece(y, np.full(100, 0.8)) == pytest.approx(0.0)  # calibrated
    y = np.array([1] * 50 + [0] * 50)
    assert ece(y, np.full(100, 0.9)) == pytest.approx(0.4)  # overconfident by 0.4


def test_entropy_peaks_at_half():
    h = entropy(np.array([0.0, 0.1, 0.5, 0.9, 1.0]))
    assert h[2] == pytest.approx(np.log(2))
    assert h[0] == pytest.approx(0, abs=1e-6) and h[1] == pytest.approx(h[3])


def test_image_reductions():
    u = np.array([[0.1, 0.9], [0.2, 0.3]])
    np.testing.assert_allclose(image_uncertainty(u, "max"), [0.9, 0.3])
    np.testing.assert_allclose(image_uncertainty(u, "mean"), [0.5, 0.25])
    y = np.array([[1, 0], [1, 1]])
    p = np.array([[0.9, 0.8], [0.2, 0.1]])
    np.testing.assert_allclose(image_brier(y, p), [0.325, 0.725])


def test_brier_separates_near_misses_from_confident_misses():
    y = np.array([[1.0], [1.0], [0.0], [0.0]])
    p = np.array([[0.15], [0.01], [0.4], [0.01]])
    loss = image_brier(y, p)
    np.testing.assert_allclose(loss, [0.7225, 0.9801, 0.16, 0.0001])
    assert loss[1] > loss[0] > loss[2] > loss[3]


def test_risk_coverage_shape_and_endpoint(rng):
    errors = rng.integers(0, 2, 100)
    cov, risk = risk_coverage(errors, rng.random(100))
    assert cov[-1] == 1.0 and len(risk) == 100
    assert risk[-1] == pytest.approx(errors.mean())  # full coverage = overall error


def test_failure_detection_perfect_vs_random(rng):
    loss = rng.random(2000)
    perfect = loss + rng.random(2000) * 0.01
    random = rng.random(2000)
    reverse = -perfect

    assert loss_spearman(loss, perfect) > 0.99
    assert loss_spearman(loss, random) == pytest.approx(0, abs=0.05)
    assert loss_spearman(loss, reverse) < -0.99
    assert aurc(loss, perfect) < aurc(loss, random) < aurc(loss, reverse)


def test_loss_spearman_nan_when_constant():
    assert np.isnan(loss_spearman(np.zeros(10), np.arange(10)))


def test_knn_distance(rng):
    train = rng.random((50, 8))
    assert np.allclose(knn_distance(train, train, k=1), 0)
    far = train[:5] + 100
    assert (knn_distance(train, far, k=3) > knn_distance(train, train[:5], k=3)).all()


def test_evaluate_and_plots_run(rng):
    y = rng.integers(0, 2, size=(100, 4))
    p = np.clip(y * 0.7 + rng.random((100, 4)) * 0.3, 0, 1)
    u = image_uncertainty(entropy(p))
    res = evaluate(y, p, u)
    for key in ("macro_auroc", "ece", "aurc", "loss_spearman", "mean_brier"):
        assert key in res
    plot_reliability(y, p)
    plot_referral(image_brier(y, p), u)



def test_missing_labels_are_skipped():
    y = np.array([[1, 0], [0, np.nan], [1, 1], [0, 0]], dtype=float)
    p = np.array([[0.9, 0.2], [0.1, 0.99], [0.8, 0.7], [0.3, 0.1]])
    filled = np.where(np.isnan(y), 0, y)
    perf = performance(y, p)
    assert perf["macro_auroc"] == 1.0
    assert np.isfinite(ece(y, p))
    assert ece(y, p) == ece(y[~np.isnan(y)], p[~np.isnan(y)])
    np.testing.assert_allclose(image_brier(y, p), [0.025, 0.01, 0.065, 0.05])
    assert image_brier(filled, p)[1] == pytest.approx(0.49505)


def test_image_with_no_observed_labels_is_dropped():
    y = np.array([[1.0], [np.nan], [0.0]])
    p = np.array([[0.2], [0.5], [0.1]])
    loss = image_brier(y, p)
    assert np.isnan(loss[1])
    unc = np.array([0.9, 0.0, 0.1])
    assert aurc(loss, unc) == aurc(loss[[0, 2]], unc[[0, 2]])
    assert loss_spearman(loss, unc) == pytest.approx(1.0)
    res = evaluate(y, p, unc)
    assert np.isfinite(res["aurc"]) and res["mean_brier"] == pytest.approx(0.325)
