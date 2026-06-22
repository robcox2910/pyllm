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
