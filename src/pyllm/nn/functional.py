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


def cross_entropy(logits, targets):
    """Score how surprised the model was by the right answer (lower = better).

    For each example the model gives a score to every possible next token. We
    turn those into probabilities (softmax) and ask: "what probability did you
    give the *correct* token?" If it was confident and right, the surprise is
    near zero; if it was confident and wrong, the surprise is huge. We average
    the surprise over all examples. Breadcrumb rule: push the predicted
    probabilities toward the true answer -- gradient is `(softmax - one_hot) / N`.
    """
    targets = np.asarray(targets)
    flat_logits = logits.data.reshape(-1, logits.data.shape[-1])
    flat_targets = targets.reshape(-1)
    n = flat_logits.shape[0]

    shifted = flat_logits - flat_logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    correct = probs[np.arange(n), flat_targets]
    loss_value = -np.log(correct).mean()
    out = Tensor(loss_value, (logits,), "cross_entropy")

    def _backward():
        grad = probs.copy()
        grad[np.arange(n), flat_targets] -= 1.0
        grad = grad / n * out.grad
        logits.grad += grad.reshape(logits.data.shape)

    out._backward = _backward
    return out
