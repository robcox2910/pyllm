"""A friendly one-call way to make PyLLM write something.

Loading a model from disk and wiring up sampling every time would be a chore, so
this is the "just give me some text" front door: call `generate_pokemon()` to
dream up creatures, or `generate_pebble()` to write Pebble code. The trained
brain is loaded once and kept warm (cached) so the second call is instant. This
is also the door PyStack knocks on to let Pebble programs use PyLLM.
"""

import numpy as np

from pyllm.checkpoint import load
from pyllm.data import PEBBLE_CHECKPOINT, POKEMON_CHECKPOINT
from pyllm.generate import generate

_CACHE = {}


def _load_cached(path):
    """Load a checkpoint once and remember it, so later calls are instant."""
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = load(path)
    return _CACHE[key]


def generate_pokemon(prompt="", max_new_tokens=60, temperature=0.8, seed=None):
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


def generate_pebble(prompt="let ", max_new_tokens=120, temperature=0.7, seed=None):
    """Write Pebble-flavoured code using the bundled Pebble brain."""
    model, tokenizer = _load_cached(PEBBLE_CHECKPOINT)
    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        rng=np.random.default_rng(seed),
    )
