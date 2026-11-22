import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr

from jepa_cxr.classifiers import fit_logistic
from jepa_cxr.compare import make_fake_splits
from jepa_cxr.laplace import fit_laplace
from jepa_cxr.uncertainty import LaplaceTunedMethod


def setup(l2=1e-2, seed=0):
    (X, Y), val, tests = make_fake_splits(
        n_train=400, n_val=200, n_test=200, dim=12, n_labels=2, seed=seed
    )
    head = fit_logistic(X, Y, l2)
    return head, X, Y, val, tests


def test_variance_larger_far_from_training_data():
    head, X, Y, _, tests = setup()
    post = fit_laplace(head, X, Y, l2=1e-2)
    near = tests["clean"][0]
    far = near * 10  # same directions, far outside the training cloud
    assert post.logit_variance(far).mean() > 10 * post.logit_variance(near).mean()


def test_probabilities_pulled_toward_half():
    head, X, Y, _, tests = setup()
    post = fit_laplace(head, X, Y, l2=1e-2)
    Xt = tests["clean"][0] * 5
    p_map = expit(head.logits(Xt))
    p_lap = post.predict_proba(Xt)
    # Laplace never more confident than MAP, and same side of 0.5
    assert (np.abs(p_lap - 0.5) <= np.abs(p_map - 0.5) + 1e-12).all()
    assert ((p_lap > 0.5) == (p_map > 0.5)).all()


def test_strong_prior_matches_map():
    head, X, Y, _, tests = setup()
    post = fit_laplace(head, X, Y, l2=1e-2, prior_scale=1e6)
    Xt = tests["clean"][0]
    # Not exact: the bias keeps its near-flat prior, so a little bias variance remains.
    np.testing.assert_allclose(post.predict_proba(Xt), expit(head.logits(Xt)), atol=5e-3)


def test_weaker_prior_means_more_variance():
    head, X, Y, _, tests = setup()
    Xt = tests["shift_2"][0]
    weak = fit_laplace(head, X, Y, l2=1e-2, prior_scale=0.01).logit_variance(Xt)
    strong = fit_laplace(head, X, Y, l2=1e-2, prior_scale=100).logit_variance(Xt)
    assert (weak > strong).all()


def test_ranking_close_to_map_on_clean_data():
    head, X, Y, _, tests = setup()
    Xt = tests["clean"][0]
    p_lap = fit_laplace(head, X, Y, l2=1e-2).predict_proba(Xt)[:, 0]
    assert spearmanr(p_lap, head.logits(Xt)[:, 0]).statistic > 0.95


def test_tuned_method_picks_from_grid():
    _, X, Y, val, _ = setup()
    m = LaplaceTunedMethod().fit(X, Y, *val)
    assert m.prior_scale in m.prior_scales

