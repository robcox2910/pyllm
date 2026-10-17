"""Tests for bundled checkpoint."""

import numpy as np

from pyllm.checkpoint import load
from pyllm.data import POKEMON_CHECKPOINT
from pyllm.generate import generate


def test_bundled_pokemon_checkpoint_exists_and_generates():
    """Bundled pokemon checkpoint exists and generates."""
    assert POKEMON_CHECKPOINT.exists(), "run: uv run python scripts/train_pokemon.py"
    model, tok = load(POKEMON_CHECKPOINT)
    out = generate(model, tok, prompt="", max_new_tokens=40, rng=np.random.default_rng(0))
    assert isinstance(out, str)
    assert len(out) > 0
    assert set(out).issubset(set(tok.stoi))
