"""The training loop: show batches over and over and nudge the weights."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from pyllm.models import LanguageModel
from pyllm.nn import cross_entropy
from pyllm.training.data import get_batch
from pyllm.training.optim import SGD, Adam


def train(
    model: LanguageModel,
    data: ArrayLike,
    steps: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-3,
    optimizer: str = "adam",
    rng: np.random.Generator | None = None,
    log_every: int = 0,
    log: Callable[[str], None] = print,
) -> list[float]:
    """Teach a model by showing it batches over and over and nudging its weights.

    This is the whole heartbeat of learning, repeated `steps` times:
    1. grab a random batch of windows (`get_batch`),
    2. let the model guess (`forward`),
    3. measure how wrong it was (`cross_entropy` -- the loss),
    4. trace the blame back to every weight (`loss.backward()`),
    5. nudge every weight a little to be less wrong (`optimizer.step()`).
    Do that enough times and the model gets good at predicting the next token.
    Returns the loss at each step so you can watch it fall.
    """
    model.train()
    params = model.parameters()
    opt: Adam | SGD = Adam(params, lr=lr) if optimizer == "adam" else SGD(params, lr=lr)
    losses: list[float] = []
    for step in range(steps):
        x, y = get_batch(data, model.block_size, batch_size, mode=model.mode, rng=rng)
        logits = model(x)
        loss = cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_value = float(loss.data)
        losses.append(loss_value)
        if log_every and step % log_every == 0:
            log(f"step {step}: loss {loss_value:.4f}")
    return losses
