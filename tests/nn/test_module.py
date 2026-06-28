import numpy as np

from pyllm.autograd import Tensor
from pyllm.nn.module import Module


class _Tiny(Module):
    def __init__(self):
        self.w = Tensor([1.0, 2.0])
        self.b = Tensor([0.0])
        self.not_a_param = 42

    def forward(self, x):
        return x + self.b


class _Nested(Module):
    def __init__(self):
        self.layer = _Tiny()
        self.heads = [_Tiny(), _Tiny()]


def test_parameters_collects_only_tensors():
    params = _Tiny().parameters()
    assert len(params) == 2
    assert all(isinstance(p, Tensor) for p in params)


def test_parameters_recurses_into_submodules_and_lists():
    # 1 submodule (2) + 2 heads (2 each) = 6 tensors
    assert len(_Nested().parameters()) == 6


def test_zero_grad_zeros_all_params():
    m = _Tiny()
    for p in m.parameters():
        p.grad += 5.0
    m.zero_grad()
    assert all(np.all(p.grad == 0.0) for p in m.parameters())


def test_call_delegates_to_forward():
    m = _Tiny()
    out = m(Tensor([10.0]))
    assert out.data.tolist() == [10.0]
