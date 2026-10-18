"""Checkpoints: freezing a trained model to a file and waking it up again."""

import json
from collections.abc import Callable
from os import PathLike
from typing import Any, cast

import numpy as np

from pyllm.models import LanguageModel, build_model
from pyllm.tokenizer import BPETokenizer, CharTokenizer, Tokenizer

_PAIR = 2
"""A BPE merge glues exactly two pieces together."""


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
    card: object = json.loads(str(data["tokenizer"])) if "tokenizer" in data else {"kind": "char"}
    vocab: list[str] = json.loads(str(data["vocab"]))
    tokenizer = _rebuild_tokenizer(card, vocab)
    if list(tokenizer.stoi) != vocab or tokenizer.vocab_size != config["vocab_size"]:
        msg = (
            f"tokenizer in checkpoint does not fit: it rebuilds {tokenizer.vocab_size} tokens, "
            f"the saved vocab lists {len(vocab)} and the model's vocab_size is "
            f"{config['vocab_size']}"
        )
        raise ValueError(msg)
    return model, tokenizer


def _rebuild_tokenizer(card: object, vocab: list[str]) -> Tokenizer:
    """Rebuild the tokenizer a checkpoint's card describes, refusing a damaged card."""
    if not isinstance(card, dict):
        msg = f"tokenizer card in checkpoint must be a dict, not {type(card).__name__}"
        raise ValueError(msg)  # noqa: TRY004 -- a bad file is a bad value, not a bad argument type
    fields = cast("dict[str, object]", card)
    kind = fields.get("kind")
    if kind == "char":
        return CharTokenizer("".join(vocab))
    if kind != "bpe":
        msg = f"unknown tokenizer kind in checkpoint: {kind!r}"
        raise ValueError(msg)
    base_chars = fields.get("base_chars")
    merges = fields.get("merges")
    if not _is_list_of(base_chars, _is_text):
        msg = "BPE tokenizer card in checkpoint needs base_chars: a list of strings"
        raise ValueError(msg)
    if not _is_list_of(merges, _is_pair_of_text):
        msg = "BPE tokenizer card in checkpoint needs merges: a list of [left, right] strings"
        raise ValueError(msg)
    pairs = [(str(left), str(right)) for left, right in cast("list[list[str]]", merges)]
    return BPETokenizer.from_merges(cast("list[str]", base_chars), pairs)


def _is_text(item: object) -> bool:
    """Say whether `item` is a string."""
    return isinstance(item, str)


def _is_pair_of_text(item: object) -> bool:
    """Say whether `item` is a two-item list of strings, like ["a", "b"]."""
    if not isinstance(item, list):
        return False
    pair = cast("list[object]", item)
    return len(pair) == _PAIR and all(isinstance(piece, str) for piece in pair)


def _is_list_of(value: object, check: Callable[[object], bool]) -> bool:
    """Say whether `value` is a list whose every item passes `check`."""
    return isinstance(value, list) and all(check(item) for item in cast("list[object]", value))
