import numpy as np

from pyllm.autograd import Tensor


def softmax(t, axis=-1):
    """Turn a row of scores into a row of probabilities that add up to 1.

    Like sharing a cake by how much each person shouted for it: louder scores get
    a bigger slice, but every slice is positive and the whole cake is shared. We
    subtract the biggest score first so the numbers never blow up (`exp` of a
    huge number overflows). Breadcrumb rule: each probability nudges its score and
    gently pushes the others down (that's the `probs * (g - sum(g*probs))` line).
    """
    shifted = t.data - t.data.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=axis, keepdims=True)
    out = Tensor(probs, (t,), "softmax")

    def _backward():
        weighted = (out.grad * probs).sum(axis=axis, keepdims=True)
        t.grad += probs * (out.grad - weighted)

    out._backward = _backward
    return out


def embedding(weight, ids):
    """Look up a row of numbers for each id — a dictionary from id to vector.

    `weight` is a table with one row per possible token. `embedding` just grabs
    the rows named by `ids` (like looking up several words in a picture
    dictionary). Breadcrumb rule: when learning, every place a row was used adds
    its blame back onto that one row (`np.add.at` does the adding-up).
    """
    ids = np.asarray(ids)
    dim = weight.data.shape[1]
    out = Tensor(weight.data[ids], (weight,), "embedding")

    def _backward():
        grad = np.zeros_like(weight.data)
        np.add.at(grad, ids.reshape(-1), out.grad.reshape(-1, dim))
        weight.grad += grad

    out._backward = _backward
    return out
