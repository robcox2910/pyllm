import numpy as np

from pyllm.training.data import get_batch


def test_sequence_batch_shapes_and_shift():
    data = np.arange(20)
    x, y = get_batch(
        data, block_size=4, batch_size=3, mode="sequence", rng=np.random.default_rng(0)
    )
    assert x.shape == (3, 4)
    assert y.shape == (3, 4)
    # y is x shifted forward by one everywhere
    for row in range(3):
        assert np.array_equal(y[row], x[row] + 1)  # data is 0,1,2,... so +1


def test_single_batch_shapes_and_target():
    data = np.arange(20)
    x, y = get_batch(
        data, block_size=4, batch_size=3, mode="single", rng=np.random.default_rng(0)
    )
    assert x.shape == (3, 4)
    assert y.shape == (3,)
    # target is the token right after the window (window ends at x[:, -1])
    for row in range(3):
        assert y[row] == x[row, -1] + 1


def test_batch_is_deterministic_with_seed():
    data = np.arange(50)
    a = get_batch(data, 4, 5, rng=np.random.default_rng(7))
    b = get_batch(data, 4, 5, rng=np.random.default_rng(7))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_batch_ids_are_integers():
    data = np.arange(20)
    x, y = get_batch(data, 4, 2, rng=np.random.default_rng(0))
    assert np.issubdtype(x.dtype, np.integer)
    assert np.issubdtype(y.dtype, np.integer)


def test_short_corpus_raises_clear_error():
    import pytest

    # Only 4 tokens but block_size=8 needs at least 9 (window + next-token answer).
    data = np.arange(4)
    with pytest.raises(ValueError, match="corpus is too short"):
        get_batch(data, block_size=8, batch_size=2, rng=np.random.default_rng(0))
