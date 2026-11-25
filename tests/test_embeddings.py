"""Tests use a tiny fake encoder, so they run without downloading any model."""

import numpy as np
import pytest
import torch

from jepa_cxr.embeddings import (
    embed,
    get_device,
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


def test_pool_mean_and_cls():
    tokens = torch.arange(2 * 4 * 8, dtype=torch.float32).view(2, 4, 8)
    torch.testing.assert_close(pool(tokens, "mean"), tokens.mean(dim=1))
    torch.testing.assert_close(pool(tokens, "cls"), tokens[:, 0])
    assert pool(torch.zeros(2, 8), "cls").shape == (2, 8)
    assert pool((tokens,), "mean").shape == (2, 8)


def test_pool_prefers_pooler_output_for_cls():
    class Out:
        last_hidden_state = torch.zeros(2, 4, 8)
        pooler_output = torch.ones(2, 8)

    assert pool(Out(), "cls").sum() == 16
    assert pool(Out(), "mean").sum() == 0  # mean ignores pooler_output


def test_embed_shape_and_batch_invariance(images):
    model = FakeEncoder().eval()
    a = embed(model, images, MEAN, STD, "mean", batch_size=2)
    b = embed(model, images, MEAN, STD, "mean", batch_size=5)
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


def test_config_validation(tmp_path):
    cfg = tmp_path / "enc.yaml"
    cfg.write_text(
        "ok:\n  hf_name: x\n  image_size: 224\n  pooling: cls\n"
        "  mean: [0.5, 0.5, 0.5]\n  std: [0.25, 0.25, 0.25]\n"
        "unset:\n  hf_name: y\n  image_size: 518\n  pooling: cls\n  mean: null\n"
        "  std: [0.25, 0.25, 0.25]\n"
        "badpool:\n  hf_name: z\n  image_size: 224\n  pooling: max\n"
        "  mean: [0.5, 0.5, 0.5]\n  std: [0.25, 0.25, 0.25]\n"
    )
    assert load_encoder_config("ok", cfg)["pooling"] == "cls"
    with pytest.raises(ValueError, match="missing"):
        load_encoder_config("unset", cfg)
    with pytest.raises(ValueError, match="pooling"):
        load_encoder_config("badpool", cfg)



def test_get_device_override(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_device().type == "cuda"
    assert get_device("auto").type == "cuda"
    assert get_device("cpu").type == "cpu"
    assert get_device("mps").type == "mps"


def test_get_device_auto_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_device().type == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert get_device().type == "cpu"


def test_get_device_rejects_unavailable_or_unknown(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        get_device("cuda")
    with pytest.raises(RuntimeError, match="mps"):
        get_device("mps")
    with pytest.raises(ValueError):
        get_device("tpu")
