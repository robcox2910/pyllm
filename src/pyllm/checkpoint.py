import json

import numpy as np

from pyllm.models import build_model
from pyllm.tokenizer import CharTokenizer


def save(path, model, tokenizer):
    """Freeze a trained model to a file so it can wake up exactly as it was.

    A checkpoint is like a photograph of the model's brain. We store three
    things: every learned number (the weights), the recipe card that says how
    the model is shaped (`config`), and the tokenizer's alphabet so text maps to
    the same numbers next time. Later `load` uses the recipe to build an empty
    model and pours the saved numbers back in.
    """
    arrays = {f"param_{i}": p.data for i, p in enumerate(model.parameters())}
    arrays["config"] = np.array(json.dumps(model.config()))
    arrays["vocab"] = np.array(json.dumps(list(tokenizer.stoi)))
    np.savez(path, **arrays)


def load(path, rng=None):
    """Wake a saved model back up: rebuild its shape, pour its numbers back in.

    The reverse of `save`. We read the recipe card, build a fresh empty model of
    the right shape, then copy each saved weight into place in the same order
    `parameters()` lists them. The tokenizer is rebuilt from the saved alphabet
    -- because our alphabet is always sorted, handing the sorted letters back to
    `CharTokenizer` reproduces the exact same letter<->number mapping.
    """
    data = np.load(path, allow_pickle=False)
    config = json.loads(str(data["config"]))
    model = build_model(config, rng=rng)
    for i, p in enumerate(model.parameters()):
        p.data = data[f"param_{i}"]
    vocab = json.loads(str(data["vocab"]))
    tokenizer = CharTokenizer("".join(vocab))
    return model, tokenizer
