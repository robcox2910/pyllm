from pyllm.api import generate_pebble, generate_pokemon


def test_generate_pokemon_returns_known_charset():
    out = generate_pokemon(prompt="", max_new_tokens=30, seed=0)
    assert isinstance(out, str) and len(out) > 0


def test_generate_pebble_starts_with_prompt():
    out = generate_pebble(prompt="let ", max_new_tokens=20, seed=0)
    assert out.startswith("let ")


def test_generation_is_reproducible_with_seed():
    a = generate_pebble(prompt="let ", max_new_tokens=30, seed=7)
    b = generate_pebble(prompt="let ", max_new_tokens=30, seed=7)
    assert a == b


def test_checkpoint_is_cached_not_reloaded():
    import pyllm.api as api

    generate_pokemon(prompt="", max_new_tokens=5, seed=1)
    from pyllm.data import POKEMON_CHECKPOINT

    assert str(POKEMON_CHECKPOINT) in api._CACHE
