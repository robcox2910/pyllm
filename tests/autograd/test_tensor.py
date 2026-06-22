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
    out = Tensor([1.0, 2.0]) + 5.0
    assert out.data.tolist() == [6.0, 7.0]


def test_add_broadcasting_reduces_grad():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
    b = Tensor([10.0, 20.0])              # shape (2,) broadcasts
    out = a + b
    out.backward()
    # b is used in both rows, so its grad sums across rows.
    assert b.grad.tolist() == [2.0, 2.0]
    assert a.grad.tolist() == [[1.0, 1.0], [1.0, 1.0]]
