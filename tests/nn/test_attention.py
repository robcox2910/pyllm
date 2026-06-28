import numpy as np

from pyllm.autograd import Tensor
from pyllm.nn.attention import Head


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
