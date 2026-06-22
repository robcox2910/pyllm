import numpy as np

from pyllm.autograd.gradcheck import numerical_grad
from pyllm.autograd.tensor import Tensor


def _check(make_output, *inputs):
    """Assert analytic grads from backward() match numerical grads."""
    out = make_output()
    out.backward()
    for t in inputs:
        approx = numerical_grad(make_output, t)
        assert np.allclose(t.grad, approx, atol=1e-4), (
            f"grad mismatch:\nanalytic={t.grad}\nnumerical={approx}"
        )


def test_gradcheck_add_mul():
    a = Tensor([1.5, -2.0, 3.0])
    b = Tensor([0.5, 4.0, -1.0])
    _check(lambda: (a * b + a).sum(), a, b)


def test_gradcheck_matmul_relu():
    a = Tensor([[1.0, -2.0], [3.0, 0.5]])
    b = Tensor([[2.0, 1.0], [-1.0, 3.0]])
    _check(lambda: (a @ b).relu().sum(), a, b)


def test_gradcheck_div_log_exp():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([2.0, 4.0, 1.0])
    _check(lambda: ((a / b).exp() + a.log()).sum(), a, b)
