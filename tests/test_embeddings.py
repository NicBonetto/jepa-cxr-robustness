"""Tests use a tiny fake encoder, so they run without downloading RadJEPA."""

import numpy as np
import pytest
import torch

from jepa_cxr.embeddings import (
    embed,
    load_embeddings,
    load_encoder_config,
    pool,
    preprocess,
    save_embeddings,
)

MEAN, STD = [0.5] * 3, [0.25] * 3


class FakeEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.proj = torch.nn.Linear(3 * 16 * 16, 4 * 8)

    def forward(self, x):
        return self.proj(x.flatten(1)).view(x.shape[0], 4, 8)


@pytest.fixture
def images():
    rng = np.random.default_rng(0)
    return [rng.random((16, 16), dtype=np.float32) for _ in range(5)]


def test_preprocess_shape(images):
    assert preprocess(images, MEAN, STD).shape == (5, 3, 16, 16)


def test_pool_handles_tokens_and_vectors():
    assert pool(torch.zeros(2, 4, 8)).shape == (2, 8)
    assert pool(torch.zeros(2, 8)).shape == (2, 8)
    assert pool((torch.zeros(2, 4, 8),)).shape == (2, 8)


def test_embed_shape_and_batch_invariance(images):
    model = FakeEncoder().eval()
    a = embed(model, images, MEAN, STD, batch_size=2)
    b = embed(model, images, MEAN, STD, batch_size=5)
    assert a.shape == (5, 8) and a.dtype == np.float32
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_save_load_roundtrip(tmp_path, images):
    emb = embed(FakeEncoder().eval(), images, MEAN, STD)
    ids = [f"img{i}" for i in range(5)]
    path = tmp_path / "out.parquet"
    save_embeddings(path, ids, emb, corruption="clean", severity=0)
    ids2, emb2 = load_embeddings(path)
    assert ids2 == ids
    np.testing.assert_array_equal(emb2, emb)


def test_config_rejects_unset_values(tmp_path):
    cfg = tmp_path / "enc.yaml"
    cfg.write_text(
        "ok:\n  hf_name: x\n  revision: null\n  image_size: 224\n"
        "  mean: [0.5, 0.5, 0.5]\n  std: [0.25, 0.25, 0.25]\n"
        "bad:\n  hf_name: y\n  image_size: null\n  mean: null\n  std: null\n"
    )
    assert load_encoder_config("ok", cfg)["image_size"] == 224
    with pytest.raises(ValueError, match="missing"):
        load_encoder_config("bad", cfg)
