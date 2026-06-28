import numpy as np


def numerical_grad(make_output, wrt, eps=1e-6):
    """Finite-difference gradient of sum(make_output().data) w.r.t. `wrt`.

    `make_output` is a function taking no arguments that builds and returns an
    output Tensor using `wrt`. We nudge each element of `wrt.data` up and down by
    `eps` and measure how the summed output changes -- the classic definition of
    a derivative. This is our independent check that backward() is correct.
    """
    grad = np.zeros_like(wrt.data)
    it = np.nditer(wrt.data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = wrt.data[idx]

        wrt.data[idx] = original + eps
        plus = float(make_output().data.sum())

        wrt.data[idx] = original - eps
        minus = float(make_output().data.sum())

        wrt.data[idx] = original
        grad[idx] = (plus - minus) / (2.0 * eps)
        it.iternext()
    return grad
