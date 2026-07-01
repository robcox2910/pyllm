import numpy as np

from pyllm.autograd import Tensor
from pyllm.autograd.gradcheck import numerical_grad
from pyllm.nn.functional import concat, cross_entropy, embedding, gelu, softmax


def test_softmax_rows_sum_to_one():
    out = softmax(Tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]), axis=-1)
    assert np.allclose(out.data.sum(axis=-1), [1.0, 1.0])


def test_softmax_is_stable_for_large_values():
    out = softmax(Tensor([[1000.0, 1000.0]]), axis=-1)
    assert np.allclose(out.data, [[0.5, 0.5]])  # no overflow


def test_softmax_gradients_check():
    a = Tensor([[0.5, -1.0, 2.0]])

    def make_output():
        # weight the probabilities so the gradient is non-trivial
        return (softmax(a, axis=-1) * Tensor([[1.0, 2.0, 3.0]])).sum()

    out = make_output()
    out.backward()
    assert np.allclose(a.grad, numerical_grad(make_output, a), atol=1e-4)


def test_embedding_looks_up_rows():
    weight = Tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    out = embedding(weight, np.array([2, 0, 1]))
    assert out.data.tolist() == [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]]


def test_embedding_backward_scatters_to_used_rows():
    weight = Tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    out = embedding(weight, np.array([0, 0, 2]))  # row 0 used twice
    out.sum().backward()
    # row 0 gets gradient 2 (used twice), row 1 zero, row 2 one
    assert weight.grad.tolist() == [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]]


def test_cross_entropy_perfect_prediction_is_near_zero():
    # huge logit on the correct class -> loss ~ 0
    logits = Tensor([[100.0, 0.0, 0.0]])
    loss = cross_entropy(logits, np.array([0]))
    assert loss.data < 1e-3


def test_cross_entropy_uniform_logits_equals_log_v():
    logits = Tensor([[0.0, 0.0, 0.0, 0.0]])  # V = 4
    loss = cross_entropy(logits, np.array([2]))
    assert np.isclose(loss.data, np.log(4))


def test_cross_entropy_gradients_check():
    logits = Tensor([[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]])
    targets = np.array([2, 0])

    def make_output():
        return cross_entropy(logits, targets)

    out = make_output()
    out.backward()
    assert np.allclose(logits.grad, numerical_grad(make_output, logits), atol=1e-4)


def test_gelu_zero_maps_to_zero():
    assert np.isclose(gelu(Tensor([0.0])).data[0], 0.0)


def test_gelu_is_close_to_identity_for_large_positive():
    assert np.isclose(gelu(Tensor([5.0])).data[0], 5.0, atol=1e-2)


def test_gelu_squashes_large_negative_toward_zero():
    assert abs(gelu(Tensor([-5.0])).data[0]) < 1e-2


def test_gelu_gradients_check():
    a = Tensor([-1.0, 0.3, 2.0])

    def make_output():
        return gelu(a).sum()

    out = make_output()
    out.backward()
    assert np.allclose(a.grad, numerical_grad(make_output, a), atol=1e-4)


def test_concat_joins_along_last_axis():
    out = concat([Tensor([[1.0, 2.0]]), Tensor([[3.0]])], axis=-1)
    assert out.data.tolist() == [[1.0, 2.0, 3.0]]


def test_concat_backward_splits_gradient():
    a = Tensor([[1.0, 2.0]])
    b = Tensor([[3.0]])

    def make_output():
        return (concat([a, b], axis=-1) * Tensor([[10.0, 20.0, 30.0]])).sum()

    out = make_output()
    out.backward()
    assert np.allclose(a.grad, numerical_grad(make_output, a), atol=1e-4)
    assert np.allclose(b.grad, numerical_grad(make_output, b), atol=1e-4)
