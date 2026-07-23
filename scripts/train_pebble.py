"""Build-time: train a tiny GPT on the Pebble corpus and save the checkpoint.

Run once with `uv run python scripts/train_pebble.py`; the resulting
`src/pyllm/data/pebble.npz` is committed so `pyllm pebble` runs instantly.
"""

import numpy as np

from pyllm.checkpoint import save
from pyllm.data import PEBBLE_CHECKPOINT, load_corpus
from pyllm.generate import generate
from pyllm.models import GPT
from pyllm.tokenizer import CharTokenizer
from pyllm.training import train


def build_and_train(rng):
    """Train a small Pebble GPT and return (model, tokenizer)."""
    text = load_corpus("pebble")
    tokenizer = CharTokenizer(text)
    data = np.array(tokenizer.encode(text))
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        block_size=24,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        rng=rng,
    )
    train(model, data, steps=1500, batch_size=32, lr=3e-3, rng=rng, log_every=150)
    return model, tokenizer


def main():
    rng = np.random.default_rng(7)
    model, tokenizer = build_and_train(rng)
    save(PEBBLE_CHECKPOINT, model, tokenizer)
    print(f"saved checkpoint to {PEBBLE_CHECKPOINT}")
    sample = generate(
        model,
        tokenizer,
        prompt="let ",
        max_new_tokens=200,
        temperature=0.7,
        rng=np.random.default_rng(0),
    )
    print("--- sample program ---")
    print(sample)


if __name__ == "__main__":
    main()
