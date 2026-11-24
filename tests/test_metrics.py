import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from jepa_cxr.metrics import (
    aurc,
    ece,
    entropy,
    error_detection_auroc,
    evaluate,
    image_errors,
    image_uncertainty,
    knn_distance,
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
    p = np.array([[0.9, 0.8], [0.2, 0.1]])  # image 0: one wrong; image 1: both wrong
    np.testing.assert_allclose(image_errors(y, p), [0.5, 1.0])


def test_risk_coverage_shape_and_endpoint(rng):
    errors = rng.integers(0, 2, 100)
    cov, risk = risk_coverage(errors, rng.random(100))
    assert cov[-1] == 1.0 and len(risk) == 100
    assert risk[-1] == pytest.approx(errors.mean())  # full coverage = overall error


def test_failure_detection_perfect_vs_random(rng):
    errors = rng.integers(0, 2, 2000)
    perfect = errors + rng.random(2000) * 0.1  # errors always more uncertain
    random = rng.random(2000)
    reverse = -perfect

    assert error_detection_auroc(errors, perfect) == 1.0
    assert error_detection_auroc(errors, random) == pytest.approx(0.5, abs=0.05)
    assert aurc(errors, perfect) < aurc(errors, random) < aurc(errors, reverse)


def test_error_detection_nan_when_no_errors():
    assert np.isnan(error_detection_auroc(np.zeros(10), np.arange(10)))


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
    for key in ("macro_auroc", "ece", "aurc", "error_detection_auroc", "mean_error"):
        assert key in res
    plot_reliability(y, p)
    plot_referral(image_errors(y, p), u)



def test_missing_labels_are_skipped():
    y = np.array([[1, 0], [0, np.nan], [1, 1], [0, 0]], dtype=float)
    p = np.array([[0.9, 0.2], [0.1, 0.99], [0.8, 0.7], [0.3, 0.1]])
    filled = np.where(np.isnan(y), 0, y)
    perf = performance(y, p)
    assert perf["macro_auroc"] == 1.0
    assert np.isfinite(ece(y, p))
    assert ece(y, p) == ece(y[~np.isnan(y)], p[~np.isnan(y)])
    assert image_errors(y, p).tolist() == [0.0, 0.0, 0.0, 0.0]
    assert image_errors(filled, p)[1] == 0.5


def test_image_with_no_observed_labels_is_dropped():
    y = np.array([[1.0], [np.nan], [0.0]])
    p = np.array([[0.2], [0.5], [0.1]])
    errors = image_errors(y, p)
    assert np.isnan(errors[1])
    unc = np.array([0.9, 0.0, 0.1])
    assert aurc(errors, unc) == aurc(errors[[0, 2]], unc[[0, 2]])
    assert error_detection_auroc(errors, unc) == 1.0
    res = evaluate(y, p, unc)
    assert np.isfinite(res["aurc"]) and res["mean_error"] == 0.5
