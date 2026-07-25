import numpy as np

from pyllm.autograd import Tensor
from pyllm.models.bigram import Bigram
from pyllm.nn import cross_entropy


def test_bigram_logits_shape():
    model = Bigram(vocab_size=5, rng=np.random.default_rng(0))
    idx = np.array([[1, 2, 3], [0, 4, 1]])  # (B, T)
    assert model(idx).shape == (2, 3, 5)


def test_bigram_mode_and_metadata():
    model = Bigram(vocab_size=5, block_size=8, rng=np.random.default_rng(0))
    assert model.mode == "sequence"
    assert model.block_size == 8
    assert model.vocab_size == 5


def test_bigram_config_roundtrips_kind():
    cfg = Bigram(vocab_size=7, block_size=4, rng=np.random.default_rng(0)).config()
    assert cfg == {"kind": "bigram", "vocab_size": 7, "block_size": 4}


def test_bigram_has_one_parameter():
    model = Bigram(vocab_size=5, rng=np.random.default_rng(0))
    assert len(model.parameters()) == 1  # just the embedding table


def test_bigram_loss_and_backward_touch_the_table():
    model = Bigram(vocab_size=5, rng=np.random.default_rng(0))
    idx = np.array([[1, 2, 3]])
    targets = np.array([[2, 3, 4]])
    loss = cross_entropy(model(idx), targets)
    loss.backward()
    assert isinstance(loss, Tensor)
    assert np.any(model.token_embedding.weight.grad != 0.0)
