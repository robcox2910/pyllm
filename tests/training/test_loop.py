import numpy as np

from pyllm.models import Bigram
from pyllm.models import MLP
from pyllm.training import train


def test_train_returns_loss_per_step():
    rng = np.random.default_rng(0)
    data = np.array([0, 1, 2, 3] * 50)
    model = Bigram(vocab_size=4, block_size=4, rng=rng)
    losses = train(model, data, steps=5, batch_size=8, rng=rng)
    assert len(losses) == 5
    assert all(isinstance(x, float) for x in losses)


def test_train_drives_loss_down_on_repetitive_data():
    rng = np.random.default_rng(0)
    data = np.array([0, 1, 2, 3] * 200)  # perfectly predictable cycle
    model = Bigram(vocab_size=4, block_size=4, rng=rng)
    losses = train(model, data, steps=300, batch_size=16, lr=0.5,
                   optimizer="sgd", rng=rng)
    assert losses[-1] < losses[0] * 0.5  # loss at least halved


def test_train_works_for_single_mode_model():
    rng = np.random.default_rng(0)
    data = np.array([0, 1, 2, 3] * 200)
    model = MLP(vocab_size=4, block_size=3, embed_dim=8, hidden_dim=16, rng=rng)
    losses = train(model, data, steps=100, batch_size=16, rng=rng)
    assert losses[-1] < losses[0]


def test_train_log_every_calls_log(capsys):
    rng = np.random.default_rng(0)
    data = np.array([0, 1, 2, 3] * 50)
    model = Bigram(vocab_size=4, block_size=4, rng=rng)
    messages = []
    train(model, data, steps=4, batch_size=8, rng=rng, log_every=2,
          log=messages.append)
    assert len(messages) == 2  # steps 0 and 2
