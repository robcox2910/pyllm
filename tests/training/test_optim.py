import numpy as np

from pyllm.autograd import Tensor
from pyllm.training.optim import SGD


def test_sgd_step_moves_downhill():
    p = Tensor([10.0])
    p.grad = np.array([2.0])  # loss increases with p -> step should decrease p
    SGD([p], lr=0.5).step()
    assert np.isclose(p.data[0], 10.0 - 0.5 * 2.0)  # 9.0


def test_sgd_minimizes_a_simple_quadratic():
    # minimize (x - 3)^2 by gradient descent; should approach x = 3
    x = Tensor([0.0])
    opt = SGD([x], lr=0.1)
    for _ in range(200):
        opt.zero_grad()
        loss = (x - 3.0) ** 2
        loss.backward()
        opt.step()
    assert np.isclose(x.data[0], 3.0, atol=1e-2)


def test_sgd_zero_grad_clears_gradients():
    p = Tensor([1.0])
    p.grad = np.array([5.0])
    SGD([p]).zero_grad()
    assert np.all(p.grad == 0.0)


from pyllm.training.optim import Adam  # noqa: E402


def test_adam_minimizes_a_simple_quadratic():
    x = Tensor([0.0])
    opt = Adam([x], lr=0.1)
    for _ in range(500):
        opt.zero_grad()
        loss = (x - 3.0) ** 2
        loss.backward()
        opt.step()
    assert np.isclose(x.data[0], 3.0, atol=1e-2)


def test_adam_first_step_size_is_about_lr():
    # On the very first step Adam's update magnitude is ~lr regardless of grad scale.
    x = Tensor([0.0])
    x.grad = np.array([1000.0])
    Adam([x], lr=0.1).step()
    assert np.isclose(abs(x.data[0]), 0.1, atol=1e-6)


def test_adam_zero_grad_clears_gradients():
    p = Tensor([1.0])
    p.grad = np.array([5.0])
    Adam([p]).zero_grad()
    assert np.all(p.grad == 0.0)
