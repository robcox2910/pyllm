import numpy as np

from pyllm.nn.embedding import Embedding


def test_embedding_output_shape():
    emb = Embedding(10, 4, rng=np.random.default_rng(0))
    ids = np.array([[1, 2, 3], [4, 5, 6]])  # (B, T)
    assert emb(ids).shape == (2, 3, 4)


def test_embedding_has_one_parameter():
    emb = Embedding(10, 4, rng=np.random.default_rng(0))
    assert len(emb.parameters()) == 1
    assert emb.weight.shape == (10, 4)


def test_embedding_trains_only_used_rows():
    emb = Embedding(5, 2, rng=np.random.default_rng(0))
    emb(np.array([1, 1, 3])).sum().backward()
    grad = emb.weight.grad
    assert np.all(grad[0] == 0.0) and np.all(grad[2] == 0.0) and np.all(grad[4] == 0.0)
    assert np.all(grad[1] != 0.0) and np.all(grad[3] != 0.0)
