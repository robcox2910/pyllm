from pyllm.nn import cross_entropy
from pyllm.training.data import get_batch
from pyllm.training.optim import SGD, Adam


def train(
    model,
    data,
    steps=1000,
    batch_size=32,
    lr=1e-3,
    optimizer="adam",
    rng=None,
    log_every=0,
    log=print,
):
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
    opt = Adam(params, lr=lr) if optimizer == "adam" else SGD(params, lr=lr)
    losses = []
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
