import numpy as np

from pyllm.checkpoint import load
from pyllm.data import PEBBLE_CHECKPOINT
from pyllm.generate import generate


def test_bundled_pebble_checkpoint_exists_and_generates():
    assert PEBBLE_CHECKPOINT.exists(), "run: uv run python scripts/train_pebble.py"
    model, tok = load(PEBBLE_CHECKPOINT)
    out = generate(
        model, tok, prompt="let ", max_new_tokens=60, rng=np.random.default_rng(0)
    )
    assert isinstance(out, str) and out.startswith("let ")
    assert set(out).issubset(set(tok.stoi))
