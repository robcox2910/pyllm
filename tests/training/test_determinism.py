"""Tests that training with a fixed seed gives the same model every time."""

import subprocess
import sys

TRAIN_AND_HASH = """
import hashlib
import numpy as np
from pyllm.models import GPT
from pyllm.training import train
rng = np.random.default_rng(0)
data = rng.integers(0, 12, size=400)
model = GPT(vocab_size=12, block_size=6, embed_dim=8, num_heads=2, num_layers=2, rng=rng)
train(model, data, steps=12, batch_size=8, lr=3e-3, rng=rng, log_every=1000, log=lambda _: None)
weights = b"".join(p.data.tobytes() for p in model.parameters())
print(hashlib.sha256(weights).hexdigest())
"""
"""A tiny seeded training run that prints a fingerprint of the trained weights."""

RUNS = 4


def _fingerprint() -> str:
    """Train in a brand-new Python process and return the weights' fingerprint."""
    result = subprocess.run(  # noqa: S603 -- runs our own fixed script
        [sys.executable, "-c", TRAIN_AND_HASH], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def test_seeded_training_gives_identical_weights_in_fresh_processes():
    """The same seed trains bit-identical weights, however Python lays out memory.

    Gradients reach a tensor from several places and get added up; if the
    order of that adding changed between runs (for example by walking a set of
    tensors, whose order depends on memory addresses), the weights would drift.
    """
    fingerprints = {_fingerprint() for _ in range(RUNS)}
    assert len(fingerprints) == 1
