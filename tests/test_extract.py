import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from jepa_cxr.embeddings import save_embeddings
from jepa_cxr.extract import (
    Job,
    build_grid,
    embed_paths,
    is_done,
    prepare,
    read_manifest,
    record,
)

ENC = {"image_size": 16, "mean": [0.5] * 3, "std": [0.25] * 3, "pooling": "mean"}


class FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.proj = torch.nn.Linear(3 * 16 * 16, 4 * 8)

    def forward(self, x):
        return self.proj(x.flatten(1)).view(x.shape[0], 4, 8)


@pytest.fixture
def cfg():
    return yaml.safe_load(open("configs/embeddings.yaml"))


@pytest.fixture
def pngs(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for i in range(5):
        path = tmp_path / f"img{i}.png"
        Image.fromarray(rng.integers(0, 256, (32, 32), dtype=np.uint8)).save(path)
        paths.append(path)
    return paths


def test_grid_from_repo_config(cfg):
    jobs = build_grid(cfg)
    # per encoder: chexpert 3 clean + 25 corrupted, vindr 1 clean + 25 corrupted
    assert len(jobs) == 2 * (3 + 25 + 1 + 25)
    corrupted_splits = {(j.dataset, j.split) for j in jobs if j.corruption != "clean"}
    assert corrupted_splits == {("chexpert", "test"), ("vindr", "test")}
    assert all(j.severity == 0 for j in jobs if j.corruption == "clean")


def test_grid_validation(cfg):
    with pytest.raises(ValueError, match="Unknown"):
        build_grid({**cfg, "corruptions": ["speckle"]})
    with pytest.raises(ValueError, match="1-5"):
        build_grid({**cfg, "severities": [0, 1]})
    with pytest.raises(ValueError, match="duplicate"):
        build_grid({**cfg, "severities": [1, 1]})


def test_job_path():
    assert Job("radjepa", "vindr", "test", "jpeg", 3).path("e").as_posix() == (
        "e/radjepa/vindr/test/jpeg_s3.parquet"
    )
    assert Job("radjepa", "vindr", "test", "clean", 0).path("e").name == "clean.parquet"


def test_prepare_clean_and_corrupted():
    img = np.random.default_rng(0).integers(0, 256, (224, 224), dtype=np.uint8)
    clean = prepare(img, "clean", 0, seed=0, size=224)
    np.testing.assert_allclose(clean, img / 255, atol=1e-6)
    a = prepare(img, "gaussian_noise", 3, seed=1, size=224)
    b = prepare(img, "gaussian_noise", 3, seed=1, size=224)
    c = prepare(img, "gaussian_noise", 3, seed=2, size=224)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c) and not np.array_equal(a, clean)
    up = prepare(img, "gaussian_noise", 3, seed=1, size=518)
    assert up.shape == (518, 518) and up.dtype == np.float32 and 0 <= up.min() <= up.max() <= 1


def test_embed_paths_chunk_invariant(pngs):
    model = FakeEncoder().eval()
    a = embed_paths(model, pngs, ENC, "gaussian_noise", 4, seed=0, batch_size=2, chunk_size=2)
    b = embed_paths(model, pngs, ENC, "gaussian_noise", 4, seed=0, batch_size=5, chunk_size=5)
    clean = embed_paths(model, pngs, ENC)
    assert a.shape == (5, 8)
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert not np.allclose(a, clean)


def test_manifest_record_and_is_done(tmp_path):
    manifest = tmp_path / "emb" / "manifest.csv"
    out = tmp_path / "emb" / "radjepa" / "vindr" / "test" / "clean.parquet"
    meta = dict(encoder="radjepa", revision="abc", dataset="vindr", split="test",
                corruption="clean", severity=0, seed=0)  # fmt: skip
    assert not is_done(read_manifest(manifest), out, manifest)

    emb = np.zeros((3, 4), dtype=np.float32)
    save_embeddings(out, ["a", "b", "c"], emb, **meta)
    record(manifest, out, emb, **meta)
    record(manifest, out, emb, **meta)  # rerun replaces, does not duplicate

    df = read_manifest(manifest)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["path"] == "radjepa/vindr/test/clean.parquet"
    assert (row["n_images"], row["dim"], len(row["sha256"])) == (3, 4, 64)
    assert is_done(df, out, manifest)
    out.unlink()
    assert not is_done(df, out, manifest)


def test_record_requires_all_metadata(tmp_path):
    out = tmp_path / "x.parquet"
    save_embeddings(out, ["a"], np.zeros((1, 2)))
    with pytest.raises(ValueError, match="missing"):
        record(tmp_path / "manifest.csv", out, np.zeros((1, 2)), encoder="radjepa")
