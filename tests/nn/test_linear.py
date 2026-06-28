import numpy as np

from pyllm.autograd import Tensor
from pyllm.autograd.gradcheck import numerical_grad
from pyllm.nn.linear import Linear


def test_linear_output_shape_batched():
    layer = Linear(4, 3, rng=np.random.default_rng(0))
    x = Tensor(np.ones((2, 5, 4)))  # (B, T, in)
    assert layer(x).shape == (2, 5, 3)


def test_linear_has_two_parameters():
    layer = Linear(4, 3, rng=np.random.default_rng(0))
    assert len(layer.parameters()) == 2  # weight + bias


def test_linear_no_bias():
    layer = Linear(4, 3, bias=False, rng=np.random.default_rng(0))
    assert layer.bias is None
    assert len(layer.parameters()) == 1


def test_linear_gradients_check():
    layer = Linear(3, 2, rng=np.random.default_rng(0))
    x = Tensor([[1.0, -2.0, 0.5]])

    def make_output():
        return layer(x).sum()

    out = make_output()
    out.backward()
    assert layer.bias is not None
    for p in (layer.weight, layer.bias, x):
        assert np.allclose(p.grad, numerical_grad(make_output, p), atol=1e-4)
