"""Checkpoints: freezing a trained model to a file and waking it up again."""

import json
from os import PathLike
from typing import Any

import numpy as np

from pyllm.models import LanguageModel, build_model
from pyllm.tokenizer import BPETokenizer, CharTokenizer, Tokenizer


def save(path: str | PathLike[str], model: LanguageModel, tokenizer: Tokenizer) -> None:
    """Freeze a trained model to a file so it can wake up exactly as it was.

    A checkpoint is like a photograph of the model's brain. We store three
    things: every learned number (the weights), the recipe card that says how
    the model is shaped (`config`), and the tokenizer's alphabet so text maps to
    the same numbers next time. Later `load` uses the recipe to build an empty
    model and pours the saved numbers back in.

    The tokenizer must be a `CharTokenizer` or a `BPETokenizer`: we note which
    kind it is (plus BPE's learned merges) so `load` rebuilds the right one.
    Any other tokenizer raises `TypeError` rather than quietly coming back as
    the wrong kind.
    """
    arrays: dict[str, Any] = {f"param_{i}": p.data for i, p in enumerate(model.parameters())}
    arrays["config"] = np.array(json.dumps(model.config()))
    arrays["vocab"] = np.array(json.dumps(list(tokenizer.stoi)))
    arrays["tokenizer"] = np.array(json.dumps(_tokenizer_card(tokenizer)))
    np.savez(path, **arrays)


def _tokenizer_card(tokenizer: Tokenizer) -> dict[str, Any]:
    """Describe a tokenizer well enough for `load` to rebuild exactly the same one."""
    if isinstance(tokenizer, BPETokenizer):
        return {
            "kind": "bpe",
            "base_chars": tokenizer.base_chars,
            "merges": [list(pair) for pair in tokenizer.merges],
        }
    if isinstance(tokenizer, CharTokenizer):
        return {"kind": "char"}
    msg = (
        f"cannot save a {type(tokenizer).__name__} in a checkpoint: "
        "only CharTokenizer and BPETokenizer can be rebuilt by load()"
    )
    raise TypeError(msg)


def load(
    path: str | PathLike[str], rng: np.random.Generator | None = None
) -> tuple[LanguageModel, Tokenizer]:
    """Wake a saved model back up: rebuild its shape, pour its numbers back in.

    The reverse of `save`. We read the recipe card, build a fresh empty model of
    the right shape, then copy each saved weight into place in the same order
    `parameters()` lists them. A BPE tokenizer is rebuilt from its saved letters
    and merges. A character tokenizer is rebuilt from the saved alphabet --
    because our alphabet is always sorted, handing the sorted letters back to
    `CharTokenizer` reproduces the exact same letter<->number mapping. Older
    checkpoints that don't say which kind they hold are character ones.
    """
    data = np.load(path, allow_pickle=False)
    config = json.loads(str(data["config"]))
    model = build_model(config, rng=rng)
    for i, p in enumerate(model.parameters()):
        p.data = data[f"param_{i}"]
    card = json.loads(str(data["tokenizer"])) if "tokenizer" in data else {"kind": "char"}
    if card["kind"] == "bpe":
        merges = [(left, right) for left, right in card["merges"]]
        return model, BPETokenizer.from_merges(card["base_chars"], merges)
    if card["kind"] == "char":
        vocab = json.loads(str(data["vocab"]))
        return model, CharTokenizer("".join(vocab))
    msg = f"unknown tokenizer kind in checkpoint: {card['kind']!r}"
    raise ValueError(msg)
