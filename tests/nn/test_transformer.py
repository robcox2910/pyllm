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


def test_dropout_defaults_to_off():
    # With dropout=0.0 the block is a deterministic computation: calling it twice
    # on the same input gives the same result, even in train mode.
    block = TransformerBlock(
        embed_dim=8, num_heads=2, block_size=16, rng=np.random.default_rng(0)
    )
    block.train()
    x = Tensor(np.random.default_rng(1).normal(size=(1, 4, 8)))
    assert np.array_equal(block(x).data, block(x).data)


def test_dropout_activates_in_train_mode_but_not_eval():
    block = TransformerBlock(
        embed_dim=8,
        num_heads=2,
        block_size=16,
        dropout=0.5,
        rng=np.random.default_rng(0),
    )
    x = Tensor(np.random.default_rng(1).normal(size=(1, 4, 8)))
    # In train mode dropout randomly zeroes signals, so two passes differ.
    block.train()
    assert not np.array_equal(block(x).data, block(x).data)
    # In eval mode dropout is off, so it is deterministic again.
    block.eval()
    assert np.array_equal(block(x).data, block(x).data)
