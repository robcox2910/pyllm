import numpy as np

from pyllm.autograd import Tensor
from pyllm.nn.dropout import Dropout


def test_dropout_eval_is_identity():
    d = Dropout(p=0.5, rng=np.random.default_rng(0))
    d.training = False
    x = Tensor([1.0, 2.0, 3.0, 4.0])
    assert d(x).data.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_dropout_has_no_parameters():
    assert Dropout(p=0.5).parameters() == []


def test_dropout_train_zeros_some_and_scales_rest():
    d = Dropout(p=0.5, rng=np.random.default_rng(0))
    x = Tensor(np.ones(1000))
    out = d(x).data
    # kept entries are scaled to 1/(1-0.5) = 2.0; dropped are 0.0
    assert set(np.unique(out)).issubset({0.0, 2.0})
    # roughly half are kept (allow slack)
    assert 350 < np.count_nonzero(out) < 650


def test_dropout_preserves_expected_value_roughly():
    d = Dropout(p=0.2, rng=np.random.default_rng(1))
    x = Tensor(np.ones(10000))
    assert np.isclose(d(x).data.mean(), 1.0, atol=0.05)
