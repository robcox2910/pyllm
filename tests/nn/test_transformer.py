import numpy as np

from pyllm.autograd import Tensor
from pyllm.nn.transformer import FeedForward, TransformerBlock


def test_feedforward_preserves_shape():
    ff = FeedForward(8, rng=np.random.default_rng(0))
    x = Tensor(np.ones((2, 5, 8)))
    assert ff(x).shape == (2, 5, 8)


def test_transformer_block_preserves_shape():
    block = TransformerBlock(
        embed_dim=8, num_heads=2, block_size=16, rng=np.random.default_rng(0)
    )
    x = Tensor(np.ones((2, 5, 8)))
    assert block(x).shape == (2, 5, 8)


def test_transformer_block_is_causal():
    block = TransformerBlock(
        embed_dim=8, num_heads=2, block_size=16, rng=np.random.default_rng(0)
    )
    rng = np.random.default_rng(2)
    base = rng.normal(size=(1, 6, 8))
    changed = base.copy()
    changed[0, -1, :] += 10.0
    out_base = block(Tensor(base)).data
    out_changed = block(Tensor(changed)).data
    assert np.allclose(out_base[0, :5], out_changed[0, :5], atol=1e-8)


def test_transformer_block_gradients_flow_to_input():
    block = TransformerBlock(
        embed_dim=8, num_heads=2, block_size=16, rng=np.random.default_rng(0)
    )
    x = Tensor(np.ones((1, 4, 8)))
    block(x).sum().backward()
    assert np.any(x.grad != 0.0)
