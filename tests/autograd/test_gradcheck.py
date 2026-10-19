"""Tests for gradcheck."""

from collections.abc import Callable

import numpy as np

from pyllm.autograd.gradcheck import numerical_grad
from pyllm.autograd.tensor import Tensor


def _check(make_output: Callable[[], Tensor], *inputs: Tensor) -> None:
    """Assert analytic grads from backward() match numerical grads."""
    out = make_output()
    out.backward()
    for t in inputs:
        approx = numerical_grad(make_output, t)
        assert np.allclose(t.grad, approx, atol=1e-4), (
            f"grad mismatch:\nanalytic={t.grad}\nnumerical={approx}"
        )


def test_gradcheck_add_mul():
    """Gradcheck add mul."""
    a = Tensor([1.5, -2.0, 3.0])
    b = Tensor([0.5, 4.0, -1.0])
    _check(lambda: (a * b + a).sum(), a, b)


def test_gradcheck_matmul_relu():
    """Gradcheck matmul relu."""
    a = Tensor([[1.0, -2.0], [3.0, 0.5]])
    b = Tensor([[2.0, 1.0], [-1.0, 3.0]])
    _check(lambda: (a @ b).relu().sum(), a, b)


def test_gradcheck_div_log_exp():
    """Gradcheck div log exp."""
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([2.0, 4.0, 1.0])
    _check(lambda: ((a / b).exp() + a.log()).sum(), a, b)


def test_gradcheck_sum_mean_tanh_pow():
    """Gradcheck sum mean tanh pow."""
    a = Tensor([[1.0, -2.0, 0.5], [3.0, 0.2, -1.0]])
    b = Tensor([0.5, 1.5, -0.7])
    _check(lambda: (((a**2).tanh()).sum(axis=0) - b).mean(), a, b)


def test_gradcheck_sub_neg():
    """Gradcheck sub neg."""
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([0.5, 0.5, 0.5])
    _check(lambda: (-(a - b)).sum(), a, b)


def test_gradcheck_reused_tensor_accumulates():
    # a appears twice: gradients must accumulate (the reason backward uses +=)
    """Gradcheck reused tensor accumulates."""
    a = Tensor([1.5, -0.5, 2.0])
    _check(lambda: (a * a + a.exp()).sum(), a)


def test_gradcheck_batched_matmul_linear_pattern():
    # x(2,2,3) @ W(3,2): proves __matmul__ unbroadcasts W.grad over the batch.
    """Gradcheck batched matmul linear pattern."""
    x = Tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[0.5, -1.0, 2.0], [1.0, 0.0, -2.0]]])
    w = Tensor([[1.0, -1.0], [0.5, 2.0], [-2.0, 0.5]])
    _check(lambda: (x @ w).sum(), x, w)


def _filled(*shape: int) -> Tensor:
    """Make a tensor of the given shape filled with distinct, non-zero numbers."""
    size = int(np.prod(shape))
    return Tensor(np.linspace(-1.5, 2.0, size).reshape(shape))


def test_gradcheck_vector_at_matrix():
    """A 1-D vector @ a matrix gets gradients shaped like each operand."""
    a, b = _filled(3), _filled(3, 4)
    _check(lambda: (a @ b).tanh().sum(), a, b)


def test_gradcheck_matrix_at_vector():
    """A matrix @ a 1-D vector gets gradients shaped like each operand."""
    a, b = _filled(2, 3), _filled(3)
    _check(lambda: (a @ b).tanh().sum(), a, b)


def test_gradcheck_vector_at_vector():
    """Two 1-D vectors make a dot product, and both get gradients."""
    a, b = _filled(3), Tensor([0.5, -1.0, 2.0])
    _check(lambda: (a @ b).tanh(), a, b)


def test_gradcheck_vector_against_batches():
    """A 1-D vector on either side of a batch of matrices gets gradients too."""
    v, batch = _filled(3), _filled(2, 3, 3)
    _check(lambda: (v @ batch).tanh().sum() + (batch @ v).tanh().sum(), v, batch)
