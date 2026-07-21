import numpy as np

from pyllm.models import GPT
from pyllm.models import build_model


def test_build_model_roundtrips_every_kind():
    rng = np.random.default_rng(0)
    originals = [
        ("bigram", {"kind": "bigram", "vocab_size": 5, "block_size": 8}),
        ("mlp", {"kind": "mlp", "vocab_size": 5, "block_size": 3,
                 "embed_dim": 8, "hidden_dim": 16}),
        ("gpt", {"kind": "gpt", "vocab_size": 5, "block_size": 8,
                 "embed_dim": 16, "num_heads": 2, "num_layers": 2}),
    ]
    for _kind, cfg in originals:
        model = build_model(cfg, rng=rng)
        assert model.config() == cfg


def test_build_model_produces_working_gpt():
    cfg = GPT(vocab_size=5, block_size=8, embed_dim=16, num_heads=2,
              num_layers=1, rng=np.random.default_rng(0)).config()
    model = build_model(cfg, rng=np.random.default_rng(0))
    assert model(np.array([[1, 2, 3]])).shape == (1, 3, 5)


def test_build_model_rejects_unknown_kind():
    import pytest
    with pytest.raises(ValueError):
        build_model({"kind": "quantum"})
