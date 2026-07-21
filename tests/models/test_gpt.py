import numpy as np

from pyllm.autograd import Tensor
from pyllm.models.gpt import GPT
from pyllm.nn import cross_entropy


def test_gpt_logits_shape():
    model = GPT(vocab_size=6, block_size=8, embed_dim=16, num_heads=2,
                num_layers=2, rng=np.random.default_rng(0))
    idx = np.array([[1, 2, 3, 4]])  # (B, T), T <= block_size
    assert model(idx).shape == (1, 4, 6)


def test_gpt_mode_and_config():
    model = GPT(vocab_size=6, block_size=8, embed_dim=16, num_heads=2,
                num_layers=2, rng=np.random.default_rng(0))
    assert model.mode == "sequence"
    assert model.config() == {
        "kind": "gpt", "vocab_size": 6, "block_size": 8,
        "embed_dim": 16, "num_heads": 2, "num_layers": 2,
    }


def test_gpt_is_causal():
    # Disturbing the LAST position must not change earlier-position logits.
    model = GPT(vocab_size=6, block_size=8, embed_dim=16, num_heads=2,
                num_layers=2, rng=np.random.default_rng(0))
    base = np.array([[1, 2, 3, 4, 5]])
    changed = base.copy()
    changed[0, -1] = 0  # change only the final token
    out_base = model(base).data
    out_changed = model(changed).data
    assert np.allclose(out_base[0, :4], out_changed[0, :4], atol=1e-8)


def test_gpt_trains_end_to_end_gradient_flow():
    model = GPT(vocab_size=6, block_size=8, embed_dim=16, num_heads=2,
                num_layers=2, rng=np.random.default_rng(0))
    idx = np.array([[1, 2, 3, 4]])
    targets = np.array([[2, 3, 4, 5]])
    loss = cross_entropy(model(idx), targets)
    loss.backward()
    assert isinstance(loss, Tensor)
    assert np.any(model.token_embedding.weight.grad != 0.0)
    assert np.any(model.head.weight.grad != 0.0)
