"""A friendly one-call way to make PyLLM write something.

Loading a model from disk and wiring up sampling every time would be a chore, so
this is the "just give me some text" front door: call `generate_pokemon()` to
dream up creatures, or `generate_pebble()` to write Pebble code. The trained
brain is loaded once and kept warm (cached) so the second call is instant. This
is also the door PyStack knocks on to let Pebble programs use PyLLM.
"""

from pathlib import Path

import numpy as np

from pyllm.checkpoint import load
from pyllm.data import PEBBLE_CHECKPOINT, POKEMON_CHECKPOINT
from pyllm.generate import generate
from pyllm.models import LanguageModel
from pyllm.pebble import referee as pebble_referee
from pyllm.tokenizer import Tokenizer

_CACHE: dict[str, tuple[LanguageModel, Tokenizer]] = {}


def _load_cached(path: Path) -> tuple[LanguageModel, Tokenizer]:
    """Load a checkpoint once and remember it, so later calls are instant."""
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = load(path)
    return _CACHE[key]


def generate_pokemon(
    prompt: str = "",
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    seed: int | None = None,
) -> str:
    """Dream up new Pokémon-ish names using the bundled Pokémon brain."""
    model, tokenizer = _load_cached(POKEMON_CHECKPOINT)
    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        rng=np.random.default_rng(seed),
    )


def generate_pebble(
    prompt: str = "let ",
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    seed: int | None = None,
    *,
    constrained: bool = False,
) -> str:
    """Write Pebble-flavoured code using the bundled Pebble brain.

    With `constrained=True` a grammar referee vetoes every illegal character, so
    what comes back is always real, parseable Pebble.
    """
    model, tokenizer = _load_cached(PEBBLE_CHECKPOINT)
    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        rng=np.random.default_rng(seed),
        referee=pebble_referee if constrained else None,
    )
