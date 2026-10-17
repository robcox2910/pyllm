"""Optimizers: the rules for stepping downhill (SGD and Adam)."""

from collections.abc import Iterable

import numpy as np

from pyllm.autograd import Tensor


class SGD:
    """Walk downhill in the direction that lowers the loss fastest.

    Imagine standing on a foggy hill wanting to reach the bottom. You can't see
    far, but you can feel which way is downhill under your feet (that's the
    gradient). SGD takes a small step that way, over and over. `lr` (learning
    rate) is your step size: too big and you overshoot the valley, too small and
    you crawl.
    """

    def __init__(self, parameters: Iterable[Tensor], lr: float = 0.1) -> None:
        """Remember which parameters to nudge and how big each step is (`lr`)."""
        self.parameters: list[Tensor] = list(parameters)
        self.lr = lr

    def step(self) -> None:
        """Take one downhill step for every parameter."""
        for p in self.parameters:
            p.data -= self.lr * p.grad

    def zero_grad(self) -> None:
        """Erase the slope readings before measuring the next ones."""
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)


class Adam:
    """A smarter downhill walker that adapts its step size per parameter.

    Plain SGD uses one step size for everything. Adam remembers two things for
    each parameter: the recent *average* slope (momentum -- keep rolling the way
    you've been going) and the recent *size* of the slope (so it takes smaller
    steps where the ground is steep and bigger steps where it's flat). This lets
    it learn faster and more steadily, which is why almost every real network
    uses it. `betas` say how long each memory lasts; `eps` avoids divide-by-zero.
    """

    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        """Set up the step size and the two memories (`m` and `v`) for each parameter."""
        self.parameters: list[Tensor] = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0

    def step(self) -> None:
        """Take one adaptive step for every parameter."""
        self.t += 1
        for i, p in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        """Erase the slope readings before measuring the next ones."""
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)
