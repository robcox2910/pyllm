import numpy as np
import pytest

from pyllm.autograd import Tensor
from pyllm.nn.attention import Head, MultiHeadAttention


def test_head_output_shape():
    head = Head(embed_dim=8, head_size=4, block_size=16, rng=np.random.default_rng(0))
    x = Tensor(np.ones((2, 5, 8)))  # (B, T, embed_dim)
    assert head(x).shape == (2, 5, 4)


def test_head_is_causal():
    # Changing the LAST time step must not change outputs at earlier steps.
    head = Head(embed_dim=8, head_size=4, block_size=16, rng=np.random.default_rng(0))
    rng = np.random.default_rng(1)
    base = rng.normal(size=(1, 5, 8))
    changed = base.copy()
    changed[0, -1, :] += 10.0  # disturb only the final position

    out_base = head(Tensor(base)).data
    out_changed = head(Tensor(changed)).data
    # earlier positions (0..3) are unchanged; only the last may differ
    assert np.allclose(out_base[0, :4], out_changed[0, :4], atol=1e-8)


def test_head_parameters_exclude_mask():
    head = Head(embed_dim=8, head_size=4, block_size=16, rng=np.random.default_rng(0))
    # 3 bias-free Linear layers, 1 weight each -> 3 params (mask is a buffer)
    assert len(head.parameters()) == 3


def test_multihead_output_shape():
    mha = MultiHeadAttention(embed_dim=8, num_heads=2, block_size=16,
                             rng=np.random.default_rng(0))
    x = Tensor(np.ones((2, 5, 8)))
    assert mha(x).shape == (2, 5, 8)


def test_multihead_requires_divisible_dim():
    with pytest.raises(AssertionError):
        MultiHeadAttention(embed_dim=8, num_heads=3, block_size=16)


def test_multihead_collects_all_head_and_proj_params():
    mha = MultiHeadAttention(embed_dim=8, num_heads=2, block_size=16,
                             rng=np.random.default_rng(0))
    # 2 heads x 3 params + proj (weight + bias) = 6 + 2 = 8
    assert len(mha.parameters()) == 8
