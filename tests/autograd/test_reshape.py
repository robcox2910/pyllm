import numpy as np

from pyllm.autograd import Tensor
from pyllm.autograd.gradcheck import numerical_grad


def test_reshape_forward_changes_shape_not_data():
    t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    out = t.reshape((3, 2))
    assert out.shape == (3, 2)
    assert out.data.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def test_reshape_flattens_last_two_axes():
    t = Tensor(np.ones((4, 2, 3)))  # (B, T, D)
    assert t.reshape((4, 6)).shape == (4, 6)


def test_reshape_backward():
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def make_output():
        return (a.reshape((6,)) * 2.0).sum()

    out = make_output()
    out.backward()
    assert np.allclose(a.grad, numerical_grad(make_output, a), atol=1e-4)
