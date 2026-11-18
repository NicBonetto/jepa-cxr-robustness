"""Extract frozen-encoder embeddings for chest x-rays from RAD-JEPA."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_encoder_config(name: str, path: str | Path = "configs/encoders.yaml") -> dict:
    cfg = yaml.safe_load(Path(path).read_text())[name]
    missing = [k for k in ("hf_name", "image_size", "mean", "std", "pooling") if cfg.get(k) is None]
    if missing:
        raise ValueError(f"Encoder {name!r} is missing {missing} in {path}")
    if cfg["pooling"] not in ("mean", "cls"):
        raise ValueError(f"pooling must be 'mean' or 'cls', got {cfg['pooling']!r}")
    return cfg

def load_encoder(
        name: str, revision: str | None = None, trust_remote_code: bool = False, device=None
    ):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(name, trust_remote_code=trust_remote_code, revision=revision)
    return model.eval().to(device or get_device())

def preprocess(images: list[np.ndarray], mean, std) -> torch.Tensor:
    x = torch.from_numpy(np.stack(images)).float().unsqueeze(1).repeat(1, 3, 1, 1)
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    return (x - m) / s

def pool(out, method: str = "mean") -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        pooled = getattr(out, "pooler_output", None)
        if method == "cls" and pooled is not None:
            return pooled
        out = out.last_hidden_state
    if out.ndim == 2:
        return out
    return out[:, 0] if method == "cls" else out.mean(dim=1)

@torch.no_grad()
def embed(
        model, images: list[np.ndarray], mean, std, pooling: str = "mean", batch_size: int = 64
    ) -> np.ndarray:
    device = next(model.parameters()).device
    chunks = []
    for i in range(0, len(images), batch_size):
        x = preprocess(images[i : i + batch_size], mean, std).to(device)
        chunks.append(pool(model(x), pooling).float().cpu().numpy())
    return np.concatenate(chunks)

def save_embeddings(path: str | Path, ids: list[str], emb: np.ndarray, **meta) -> None:
    df = pd.DataFrame({"image_id": ids, "embedding": list(emb.astype(np.float32))})
    for k, v in meta.items():
        df[k] = v
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def load_embeddings(path: str | Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_parquet(path)
    return df["image_id"].tolist(), np.stack(df["embedding"].to_numpy()).astype(np.float32)
