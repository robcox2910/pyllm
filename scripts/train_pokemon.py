"""Build-time: train a tiny GPT on the Pokémon corpus and save the checkpoint.

Run once with `uv run python scripts/train_pokemon.py`; the resulting
`src/pyllm/data/pokemon.npz` is committed so the shipped CLI generates instantly.
"""

import sys

import numpy as np

from pyllm.checkpoint import save
from pyllm.data import POKEMON_CHECKPOINT, load_corpus
from pyllm.generate import generate
from pyllm.models import GPT
from pyllm.tokenizer import CharTokenizer
from pyllm.training import train


def _say(message: str) -> None:
    """Show a message on the screen."""
    sys.stdout.write(message + "\n")


def build_and_train(rng: np.random.Generator) -> tuple[GPT, CharTokenizer]:
    """Train a small Pokémon GPT and return (model, tokenizer)."""
    text = load_corpus("pokemon")
    tokenizer = CharTokenizer(text)
    data = np.array(tokenizer.encode(text))
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        block_size=16,
        embed_dim=48,
        num_heads=4,
        num_layers=2,
        rng=rng,
    )
    train(model, data, steps=2500, batch_size=32, lr=3e-3, rng=rng, log_every=250)
    return model, tokenizer


def main() -> None:
    """Run the build step and report what happened."""
    rng = np.random.default_rng(1234)
    model, tokenizer = build_and_train(rng)
    save(POKEMON_CHECKPOINT, model, tokenizer)
    _say(f"saved checkpoint to {POKEMON_CHECKPOINT}")
    sample = generate(
        model,
        tokenizer,
        prompt="",
        max_new_tokens=120,
        temperature=0.8,
        rng=np.random.default_rng(0),
    )
    _say("--- sample dream ---")
    _say(sample)


if __name__ == "__main__":
    main()
