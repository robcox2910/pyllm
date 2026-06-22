import numpy as np

from pyllm.autograd import Tensor


def test_zero_grad_resets():
    a = Tensor([1.0, 2.0])
    (a.sum()).backward()
    assert a.grad.tolist() == [1.0, 1.0]
    a.zero_grad()
    assert a.grad.tolist() == [0.0, 0.0]


def test_linear_regression_converges():
    # Learn y = 2x + 1 from data, using only the autograd engine.
    rng = np.random.default_rng(0)
    xs = rng.uniform(-1.0, 1.0, size=(20, 1))
    ys = 2.0 * xs + 1.0

    w = Tensor([[0.0]])
    b = Tensor([0.0])
    x = Tensor(xs)
    target = Tensor(ys)

    lr = 0.1
    losses = []
    for _ in range(200):
        w.zero_grad()
        b.zero_grad()
        pred = x @ w + b
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        w.data -= lr * w.grad
        b.data -= lr * b.grad
        losses.append(float(loss.data))

    assert losses[-1] < 1e-3
    assert np.isclose(w.data[0, 0], 2.0, atol=0.05)
    assert np.isclose(b.data[0], 1.0, atol=0.05)
