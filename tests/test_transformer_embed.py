
import numpy as np
import pytest
import torch
import tempfile

from transformer.encoder import MultiScaleEncoder
from transformer.embed import load_encoder


def test_load_encoder_returns_none_when_missing(tmp_path):
    result = load_encoder(path=tmp_path / "nonexistent.pt")
    assert result is None


def test_load_encoder_returns_model(tmp_path):
    checkpoint = tmp_path / "finetuned.pt"
    model = MultiScaleEncoder()
    torch.save(model.state_dict(), checkpoint)
    loaded = load_encoder(path=checkpoint)
    assert isinstance(loaded, MultiScaleEncoder)


def test_load_encoder_is_eval_mode(tmp_path):
    checkpoint = tmp_path / "finetuned.pt"
    model = MultiScaleEncoder()
    torch.save(model.state_dict(), checkpoint)
    loaded = load_encoder(path=checkpoint)
    assert not loaded.training


def test_loaded_encoder_embed_shape(tmp_path):
    checkpoint = tmp_path / "finetuned.pt"
    model = MultiScaleEncoder()
    torch.save(model.state_dict(), checkpoint)
    loaded = load_encoder(path=checkpoint)
    x = np.random.randn(720, 5).astype(np.float32)
    emb = loaded.embed(x)
    assert emb.shape == (64,)
