import numpy as np

from pyllm.models.mlp import MLP
from pyllm.nn import cross_entropy


def test_mlp_logits_shape_is_one_prediction_per_window():
    model = MLP(vocab_size=5, block_size=3, embed_dim=8, hidden_dim=16,
                rng=np.random.default_rng(0))
    idx = np.array([[1, 2, 3], [0, 4, 1]])  # (B, block_size)
    assert model(idx).shape == (2, 5)  # one next-token distribution per row


def test_mlp_mode_and_metadata():
    model = MLP(vocab_size=5, block_size=3, rng=np.random.default_rng(0))
    assert model.mode == "single"
    assert model.block_size == 3
    assert model.vocab_size == 5


def test_mlp_config():
    cfg = MLP(vocab_size=5, block_size=3, embed_dim=8, hidden_dim=16,
              rng=np.random.default_rng(0)).config()
    assert cfg == {
        "kind": "mlp", "vocab_size": 5, "block_size": 3,
        "embed_dim": 8, "hidden_dim": 16,
    }


def test_mlp_has_embedding_and_two_linears():
    model = MLP(vocab_size=5, block_size=3, embed_dim=8, hidden_dim=16,
                rng=np.random.default_rng(0))
    # embedding(1) + fc1(weight+bias) + fc2(weight+bias) = 5
    assert len(model.parameters()) == 5


def test_mlp_learns_gradients_flow():
    model = MLP(vocab_size=5, block_size=3, embed_dim=8, hidden_dim=16,
                rng=np.random.default_rng(0))
    idx = np.array([[1, 2, 3]])
    targets = np.array([4])  # (B,) single next token
    loss = cross_entropy(model(idx), targets)
    loss.backward()
    assert np.any(model.fc1.weight.grad != 0.0)
