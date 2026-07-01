import numpy as np

from pyllm.autograd import Tensor
from pyllm.autograd.gradcheck import numerical_grad


def test_transpose_forward_swaps_last_two_axes():
    t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    assert t.transpose().shape == (3, 2)
    assert t.transpose().data.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]


def test_transpose_batched():
    t = Tensor(np.ones((4, 2, 3)))  # (B, T, D)
    assert t.transpose().shape == (4, 3, 2)


def test_transpose_backward():
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = (a.transpose() * 2.0).sum()
    out.backward()
    approx = numerical_grad(lambda: (a.transpose() * 2.0).sum(), a)
    assert np.allclose(a.grad, approx, atol=1e-4)
