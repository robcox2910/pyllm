"""Bundled training corpora and the shipped Pokémon checkpoint."""

from pathlib import Path

CORPUS_DIR = Path(__file__).parent
POKEMON_CHECKPOINT = CORPUS_DIR / "pokemon.npz"


def load_corpus(name="pokemon"):
    """Read one of the bundled text corpora and hand back its text.

    A corpus is just a big text file we train on. The Pokémon one is a list of
    real Pokémon names -- small enough to train on a laptop in minutes, but full
    of the sounds and shapes that make a name feel Pokémon-ish, so the model can
    learn to dream up new ones.
    """
    return (CORPUS_DIR / f"{name}_corpus.txt").read_text(encoding="utf-8")
