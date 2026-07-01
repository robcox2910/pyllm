import numpy as np

from pyllm.autograd import Tensor
from pyllm.autograd.gradcheck import numerical_grad
from pyllm.nn.normalization import LayerNorm


def test_layernorm_outputs_zero_mean_unit_var():
    ln = LayerNorm(4)
    x = Tensor([[1.0, 2.0, 3.0, 10.0]])
    out = ln(x).data
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(out.var(axis=-1), 1.0, atol=1e-3)


def test_layernorm_has_two_parameters():
    assert len(LayerNorm(4).parameters()) == 2


def test_layernorm_gradients_check():
    ln = LayerNorm(3)
    x = Tensor([[1.0, -2.0, 0.5], [3.0, 0.0, 1.0]])

    def make_output():
        return ln(x).sum()

    out = make_output()
    out.backward()
    for p in (ln.gamma, ln.beta, x):
        assert np.allclose(p.grad, numerical_grad(make_output, p), atol=1e-4)
