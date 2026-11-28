import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from jepa_cxr.embeddings import save_embeddings
from jepa_cxr.extract import record
from jepa_cxr.stability import (
    align,
    cosine_distance,
    knn_overlap,
    neighbours,
    plot_severity,
    stability,
    stability_table,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def emb(rng):
    return rng.normal(size=(60, 8)).astype(np.float32)


# cosine_distance


def test_cosine_identical_scaled_and_opposite(emb):
    assert np.allclose(cosine_distance(emb, emb), 0, atol=1e-6)
    assert np.allclose(cosine_distance(emb, 3 * emb), 0, atol=1e-6)  # scale-invariant
    assert np.allclose(cosine_distance(emb, -emb), 2, atol=1e-6)


def test_cosine_orthogonal():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    b = np.array([[0.0, 5.0], [3.0, 0.0]])
    assert np.allclose(cosine_distance(a, b), 1)


def test_cosine_rejects_zero_norm_and_shape_mismatch(emb):
    z = emb.copy()
    z[0] = 0
    with pytest.raises(ValueError, match="Zero-norm"):
        cosine_distance(emb, z)
    with pytest.raises(ValueError, match="Shapes"):
        cosine_distance(emb, emb[:-1])


# align


def test_align_reorders_by_id(emb):
    ids = [f"img{i}" for i in range(len(emb))]
    perm = np.random.default_rng(1).permutation(len(emb))
    a, b = align(ids, emb, [ids[i] for i in perm], emb[perm])
    assert np.array_equal(a, b)


def test_align_rejects_mismatched_or_duplicate_ids(emb):
    ids = [f"img{i}" for i in range(len(emb))]
    with pytest.raises(ValueError, match="differ"):
        align(ids, emb, ids[:-1] + ["other"], emb)
    with pytest.raises(ValueError, match="Duplicate"):
        align(ids, emb, ids[:-1] + [ids[0]], emb)


# neighbours / knn_overlap


def test_neighbours_exclude_self_and_are_sorted(emb):
    idx, dist = neighbours(emb, k=5)
    assert idx.shape == dist.shape == (len(emb), 5)
    assert not np.any(idx == np.arange(len(emb))[:, None])
    assert np.all(np.diff(dist, axis=1) >= -1e-9)


def test_neighbours_known_layout():
    # Two tight clusters of directions: each point's nearest neighbour is in its own cluster.
    a = np.array([[1, 0.01], [1, 0.02], [1, 0.03], [0.01, 1], [0.02, 1], [0.03, 1]])
    idx, _ = neighbours(a, k=2)
    assert all(set(idx[i]) <= {0, 1, 2} for i in range(3))
    assert all(set(idx[i]) <= {3, 4, 5} for i in range(3, 6))


def test_neighbours_with_duplicate_rows():
    a = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    idx, _ = neighbours(a, k=2)
    assert not np.any(idx == np.arange(4)[:, None])
    assert idx[0, 0] == 1 and idx[1, 0] == 0


def test_neighbours_rejects_bad_k(emb):
    with pytest.raises(ValueError, match="k must be"):
        neighbours(emb, k=len(emb))
    with pytest.raises(ValueError, match="k must be"):
        neighbours(emb, k=0)


def test_knn_overlap_values():
    a = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
    b = np.array([[4, 3, 2, 1], [1, 9, 8, 7]])  # order does not matter
    assert np.allclose(knn_overlap(a, b), [1.0, 0.25])


# stability


def test_identity_is_perfectly_stable(emb):
    res = stability(emb, emb.copy(), k=5)
    assert res["n_images"] == len(emb)
    assert res["cosine_mean"] == pytest.approx(0, abs=1e-6)
    assert res["rel_shift_median"] == pytest.approx(0, abs=1e-5)
    assert res["left_neighbourhood"] == 0
    assert res["knn_overlap"] == 1


def test_stability_degrades_with_noise(emb, rng):
    small = stability(emb, emb + 0.05 * rng.normal(size=emb.shape), k=5)
    large = stability(emb, emb + 2.0 * rng.normal(size=emb.shape), k=5)
    assert small["cosine_mean"] < large["cosine_mean"]
    assert small["rel_shift_median"] < large["rel_shift_median"]
    assert small["knn_overlap"] > large["knn_overlap"]


def test_unrelated_embeddings_give_chance_overlap(rng):
    n, k = 400, 10
    a, b = rng.normal(size=(n, 16)), rng.normal(size=(n, 16))
    res = stability(a, b, k=k)
    assert res["knn_overlap"] == pytest.approx(k / (n - 1), abs=0.02)
    assert res["left_neighbourhood"] > 0.9


def test_rel_shift_is_scale_free(emb, rng):
    # Rotating the whole space changes nothing; relative shift depends on geometry only.
    noisy = emb + 0.3 * rng.normal(size=emb.shape)
    q, _ = np.linalg.qr(rng.normal(size=(emb.shape[1],) * 2))
    a, b = stability(emb, noisy, k=5), stability(emb @ q, noisy @ q, k=5)
    for key in ("cosine_mean", "rel_shift_median", "knn_overlap"):
        assert a[key] == pytest.approx(b[key], abs=1e-5)


def test_stability_reuses_clean_neighbours(emb, rng):
    noisy = emb + 0.3 * rng.normal(size=emb.shape)
    assert stability(emb, noisy, 5, neighbours(emb, 5)) == stability(emb, noisy, 5)


# stability_table / plot


META = dict(revision="r1", dataset="chexpert", split="test", seed=0)


def _write(tmp_path, enc, corruption, severity, ids, emb, **over):
    name = "clean" if corruption == "clean" else f"{corruption}_s{severity}"
    path = tmp_path / enc / "chexpert" / "test" / f"{name}.parquet"
    meta = META | dict(encoder=enc, corruption=corruption, severity=severity) | over
    save_embeddings(path, ids, emb, **meta)
    record(tmp_path / "manifest.csv", path, emb, **meta)


@pytest.fixture
def grid(tmp_path, emb, rng):
    ids = [f"img{i}" for i in range(len(emb))]
    for enc in ("radjepa", "rad_dino"):
        _write(tmp_path, enc, "clean", 0, ids, emb)
        for s, scale in enumerate((0.05, 0.5, 2.0), start=1):
            perm = rng.permutation(len(emb))  # stored order differs from the clean file
            noisy = emb + scale * rng.normal(size=emb.shape)
            _write(tmp_path, enc, "gaussian_noise", s, [ids[i] for i in perm], noisy[perm])
    return tmp_path


def test_table_rows_and_monotone_severity(grid):
    t = stability_table(grid / "manifest.csv", k=5)
    assert len(t) == 6
    assert set(t["encoder"]) == {"radjepa", "rad_dino"}
    for _, g in t.groupby("encoder"):
        g = g.sort_values("severity")
        assert g["cosine_mean"].is_monotonic_increasing
        assert g["knn_overlap"].is_monotonic_decreasing


def test_table_filters(grid):
    t = stability_table(grid / "manifest.csv", k=5, encoders=["radjepa"])
    assert set(t["encoder"]) == {"radjepa"}
    with pytest.raises(ValueError, match="No corrupted"):
        stability_table(grid / "manifest.csv", k=5, datasets=["vindr"])


def test_table_requires_clean_counterpart(tmp_path, emb):
    ids = [f"img{i}" for i in range(len(emb))]
    _write(tmp_path, "radjepa", "gaussian_noise", 1, ids, emb)
    with pytest.raises(ValueError, match="one clean file"):
        stability_table(tmp_path / "manifest.csv", k=5)


def test_table_rejects_seed_mismatch(tmp_path, emb):
    ids = [f"img{i}" for i in range(len(emb))]
    _write(tmp_path, "radjepa", "clean", 0, ids, emb)
    _write(tmp_path, "radjepa", "gaussian_noise", 1, ids, emb, seed=1)
    with pytest.raises(ValueError, match="seed or revision"):
        stability_table(tmp_path / "manifest.csv", k=5)


def test_table_rejects_missing_images(tmp_path, emb):
    ids = [f"img{i}" for i in range(len(emb))]
    _write(tmp_path, "radjepa", "clean", 0, ids, emb)
    _write(tmp_path, "radjepa", "gaussian_noise", 1, ids[:-1], emb[:-1])
    with pytest.raises(ValueError, match="differ"):
        stability_table(tmp_path / "manifest.csv", k=5)


def test_plot_severity(grid):
    fig = plot_severity(stability_table(grid / "manifest.csv", k=5))
    assert len(fig.axes) == 2 * 3
