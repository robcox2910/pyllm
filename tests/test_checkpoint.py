"""Tests for checkpoint."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyllm.checkpoint import load, save
from pyllm.generate import generate
from pyllm.models import GPT
from pyllm.tokenizer import BPETokenizer, CharTokenizer


def test_save_load_roundtrips_config_and_weights(tmp_path: Path):
    """Save load roundtrips config and weights."""
    text = "pikachu"
    tok = CharTokenizer(text)
    model = GPT(
        vocab_size=tok.vocab_size,
        block_size=8,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        rng=np.random.default_rng(0),
    )
    path = tmp_path / "model.npz"
    save(path, model, tok)
    reloaded, _ = load(path)
    assert reloaded.config() == model.config()
    for a, b in zip(model.parameters(), reloaded.parameters(), strict=True):
        assert np.allclose(a.data, b.data)


def test_reloaded_tokenizer_matches(tmp_path: Path):
    """Reloaded tokenizer matches."""
    text = "bulbasaur"
    tok = CharTokenizer(text)
    model = GPT(
        vocab_size=tok.vocab_size,
        block_size=8,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        rng=np.random.default_rng(0),
    )
    path = tmp_path / "m.npz"
    save(path, model, tok)
    _, reloaded_tok = load(path)
    assert reloaded_tok.stoi == tok.stoi
    assert reloaded_tok.decode(reloaded_tok.encode("bulba")) == "bulba"


def test_reloaded_model_generates_identically(tmp_path: Path):
    """Reloaded model generates identically."""
    text = "charmander"
    tok = CharTokenizer(text)
    model = GPT(
        vocab_size=tok.vocab_size,
        block_size=8,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        rng=np.random.default_rng(0),
    )
    path = tmp_path / "m.npz"
    save(path, model, tok)
    reloaded, reloaded_tok = load(path)
    a = generate(model, tok, prompt="c", max_new_tokens=10, rng=np.random.default_rng(3))
    b = generate(
        reloaded,
        reloaded_tok,
        prompt="c",
        max_new_tokens=10,
        rng=np.random.default_rng(3),
    )
    assert a == b


def _tiny_gpt(vocab_size: int) -> GPT:
    """Build a tiny GPT for checkpoint tests."""
    return GPT(
        vocab_size=vocab_size,
        block_size=8,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        rng=np.random.default_rng(0),
    )


def test_bpe_tokenizer_round_trips(tmp_path: Path):
    """A BPE checkpoint loads back as the same BPE tokenizer, not a CharTokenizer."""
    text = "abababcabcabc"
    tok = BPETokenizer()
    tok.train(text, num_merges=3)
    path = tmp_path / "bpe.npz"
    save(path, _tiny_gpt(tok.vocab_size), tok)
    _, reloaded_tok = load(path)
    assert isinstance(reloaded_tok, BPETokenizer)
    assert reloaded_tok.merges == tok.merges
    assert reloaded_tok.stoi == tok.stoi
    assert reloaded_tok.encode(text) == tok.encode(text)


class _MysteryTokenizer:
    """A tokenizer the checkpoint format does not know how to store."""

    def __init__(self) -> None:
        """Know just two letters."""
        self.stoi = {"a": 0, "b": 1}

    @property
    def vocab_size(self) -> int:
        """Return how many letters are known."""
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """Turn text into numbers."""
        return [self.stoi[c] for c in text]

    def decode(self, ids: Iterable[int]) -> str:
        """Turn numbers into text."""
        return "".join("ab"[i] for i in ids)


def test_save_refuses_an_unknown_tokenizer(tmp_path: Path):
    """Saving with a tokenizer we can't rebuild fails loudly instead of loading wrong later."""
    tok = _MysteryTokenizer()
    with pytest.raises(TypeError, match="_MysteryTokenizer"):
        save(tmp_path / "m.npz", _tiny_gpt(tok.vocab_size), tok)


def test_old_checkpoint_without_tokenizer_kind_loads_as_char(tmp_path: Path):
    """A checkpoint written before the tokenizer kind was stored still loads as characters."""
    tok = CharTokenizer("pikachu")
    model = _tiny_gpt(tok.vocab_size)
    path = tmp_path / "old.npz"
    arrays: dict[str, Any] = {f"param_{i}": p.data for i, p in enumerate(model.parameters())}
    arrays["config"] = np.array(json.dumps(model.config()))
    arrays["vocab"] = np.array(json.dumps(list(tok.stoi)))
    np.savez(path, **arrays)
    _, reloaded_tok = load(path)
    assert isinstance(reloaded_tok, CharTokenizer)
    assert reloaded_tok.stoi == tok.stoi


def test_load_refuses_an_unknown_tokenizer_kind(tmp_path: Path):
    """A checkpoint naming a tokenizer kind we don't know fails loudly."""
    tok = CharTokenizer("pikachu")
    path = tmp_path / "m.npz"
    save(path, _tiny_gpt(tok.vocab_size), tok)
    arrays = dict(np.load(path, allow_pickle=False))
    arrays["tokenizer"] = np.array(json.dumps({"kind": "wordpiece"}))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="wordpiece"):
        load(path)


DROPOUT = 0.25


def test_gpt_dropout_survives_a_round_trip(tmp_path: Path):
    """A GPT saved with dropout switched on loads back with the same dropout."""
    tok = CharTokenizer("pikachu")
    model = GPT(
        vocab_size=tok.vocab_size,
        block_size=8,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=DROPOUT,
        rng=np.random.default_rng(0),
    )
    path = tmp_path / "m.npz"
    save(path, model, tok)
    reloaded, _ = load(path)
    assert isinstance(reloaded, GPT)
    assert reloaded.dropout == DROPOUT
    assert reloaded.blocks[0].drop1.p == DROPOUT


def test_old_gpt_checkpoint_without_dropout_loads_with_dropout_off(tmp_path: Path):
    """A GPT checkpoint written before dropout was saved loads with dropout off."""
    tok = CharTokenizer("pikachu")
    model = _tiny_gpt(tok.vocab_size)
    config = model.config()
    config.pop("dropout", None)
    path = tmp_path / "old.npz"
    arrays: dict[str, Any] = {f"param_{i}": p.data for i, p in enumerate(model.parameters())}
    arrays["config"] = np.array(json.dumps(config))
    arrays["vocab"] = np.array(json.dumps(list(tok.stoi)))
    np.savez(path, **arrays)
    reloaded, _ = load(path)
    assert isinstance(reloaded, GPT)
    assert reloaded.dropout == 0.0
