import numpy as np

from pyllm.autograd.tensor import Tensor


def test_wraps_data_as_float64():
    t = Tensor([1, 2, 3])
    assert t.data.dtype == np.float64
    assert t.data.tolist() == [1.0, 2.0, 3.0]


def test_grad_starts_at_zero_same_shape():
    t = Tensor([[1, 2], [3, 4]])
    assert t.grad.shape == (2, 2)
    assert np.all(t.grad == 0.0)


def test_shape_property():
    assert Tensor([[1, 2, 3]]).shape == (1, 3)


def test_repr_mentions_tensor():
    assert "Tensor" in repr(Tensor([1.0]))


def test_add_forward():
    out = Tensor([1.0, 2.0]) + Tensor([10.0, 20.0])
    assert out.data.tolist() == [11.0, 22.0]


def test_add_backward_passes_grad_to_both():
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])
    out = a + b
    out.backward()
    assert a.grad.tolist() == [1.0, 1.0]
    assert b.grad.tolist() == [1.0, 1.0]


def test_add_scalar_on_right():
    a = Tensor([1.0, 2.0])
    out = a + 5.0
    assert out.data.tolist() == [6.0, 7.0]
    out.backward()
    assert a.grad.tolist() == [1.0, 1.0]


def test_add_broadcasting_reduces_grad():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
    b = Tensor([10.0, 20.0])              # shape (2,) broadcasts
    out = a + b
    out.backward()
    # b is used in both rows, so its grad sums across rows.
    assert b.grad.tolist() == [2.0, 2.0]
    assert a.grad.tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_mul_forward():
    out = Tensor([2.0, 3.0]) * Tensor([4.0, 5.0])
    assert out.data.tolist() == [8.0, 15.0]


def test_mul_backward_uses_other_operand():
    a = Tensor([2.0, 3.0])
    b = Tensor([4.0, 5.0])
    out = a * b
    out.backward()
    assert a.grad.tolist() == [4.0, 5.0]  # d(a*b)/da = b
    assert b.grad.tolist() == [2.0, 3.0]  # d(a*b)/db = a


def test_mul_scalar_on_right():
    out = Tensor([2.0, 3.0]) * 10.0
    assert out.data.tolist() == [20.0, 30.0]


def test_matmul_forward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    out = a @ b
    assert out.data.tolist() == [[19.0, 22.0], [43.0, 50.0]]


def test_matmul_backward_shapes_and_values():
    a = Tensor([[1.0, 2.0, 3.0]])      # (1, 3)
    b = Tensor([[1.0], [1.0], [1.0]])  # (3, 1)
    out = a @ b                         # (1, 1)
    out.backward()
    # d(a@b)/da = out.grad @ b.T ; with out.grad = ones((1,1)) -> b.T = [[1,1,1]]
    assert a.grad.tolist() == [[1.0, 1.0, 1.0]]
    # d(a@b)/db = a.T @ out.grad -> [[1],[2],[3]]
    assert b.grad.tolist() == [[1.0], [2.0], [3.0]]
