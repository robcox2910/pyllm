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


def test_gradcheck_sum_mean_tanh_pow():
    a = Tensor([[1.0, -2.0, 0.5], [3.0, 0.2, -1.0]])
    b = Tensor([0.5, 1.5, -0.7])
    _check(lambda: (((a**2).tanh()).sum(axis=0) - b).mean(), a, b)


def test_gradcheck_sub_neg():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([0.5, 0.5, 0.5])
    _check(lambda: (-(a - b)).sum(), a, b)


def test_gradcheck_reused_tensor_accumulates():
    # a appears twice: gradients must accumulate (the reason backward uses +=)
    a = Tensor([1.5, -0.5, 2.0])
    _check(lambda: (a * a + a.exp()).sum(), a)


def test_gradcheck_batched_matmul_linear_pattern():
    # x(2,2,3) @ W(3,2): proves __matmul__ unbroadcasts W.grad over the batch.
    x = Tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[0.5, -1.0, 2.0], [1.0, 0.0, -2.0]]]
    )
    w = Tensor([[1.0, -1.0], [0.5, 2.0], [-2.0, 0.5]])
    _check(lambda: (x @ w).sum(), x, w)
