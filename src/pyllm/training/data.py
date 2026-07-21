import numpy as np


def get_batch(data, block_size, batch_size, mode="sequence", rng=None):
    """Grab a handful of random windows of text to learn from this step.

    Training on the whole corpus at once would be slow, so each step we grab a
    small random sample -- a "batch" -- of little windows cut from the text.
    `x` is what the model reads; `y` is the answer it should predict.

    - "sequence" (bigram, GPT): the model predicts the next token at *every*
      position, so `y` is just `x` slid one character to the right.
    - "single" (MLP): the model reads a whole window and predicts the *one*
      character that comes right after it.
    """
    if rng is None:
        rng = np.random.default_rng()
    data = np.asarray(data)
    max_start = len(data) - block_size - 1
    starts = rng.integers(0, max_start + 1, size=batch_size)
    x = np.stack([data[s:s + block_size] for s in starts])
    if mode == "sequence":
        y = np.stack([data[s + 1:s + 1 + block_size] for s in starts])
    else:  # "single"
        y = np.array([data[s + block_size] for s in starts])
    return x.astype(np.int64), y.astype(np.int64)
