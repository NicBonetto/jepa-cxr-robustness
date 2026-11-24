import numpy as np

from jepa_cxr.compare import make_fake_splits
from jepa_cxr.uncertainty import KNNMethod, SoftmaxMethod


def splits():
    return make_fake_splits(n_train=500, n_val=200, n_test=200, dim=12, n_labels=2)


def test_far_inputs_get_higher_uncertainty():
    train, val, tests = splits()
    # Euclidean only: cosine distance ignores magnitude, so "far" there means a different
    # direction, not a larger offset.
    m = KNNMethod(metrics=("euclidean",)).fit(*train, *val)
    X = tests["clean"][0]
    _, near = m.predict(X)
    _, far = m.predict(X + 20.0)  # pushed far outside the training cloud
    assert (far > near).all()


def test_probabilities_match_softmax():
    train, val, tests = splits()
    X = tests["clean"][0]
    knn_probs, _ = KNNMethod().fit(*train, *val).predict(X)
    soft_probs, _ = SoftmaxMethod().fit(*train, *val).predict(X)
    np.testing.assert_array_equal(knn_probs, soft_probs)


def test_selected_settings_come_from_grid():
    train, val, _ = splits()
    m = KNNMethod(k_grid=(1, 5), metrics=("euclidean",)).fit(*train, *val)
    assert m.k in (1, 5) and m.metric == "euclidean"


def test_k_larger_than_train_set_is_skipped():
    train, val, tests = make_fake_splits(n_train=20, n_val=30, n_test=10, dim=4, n_labels=1)
    m = KNNMethod(k_grid=(5, 50)).fit(*train, *val)
    assert m.k == 5
    assert m.predict(tests["clean"][0])[1].shape == (10,)

