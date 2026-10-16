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


def test_constrained_pebble_is_always_a_finished_program():
    from pyllm.api import generate_pebble
    from pyllm.pebble import complete

    for seed in range(3):
        text = generate_pebble(max_new_tokens=150, seed=seed, constrained=True)
        assert complete(text), text


def test_constrained_pebble_parses_under_the_real_parser():
    import pytest

    pytest.importorskip("pebble")
    from pyllm.api import generate_pebble
    from pyllm.pebble import parse_rate

    programs = []
    for seed in range(4):
        text = generate_pebble(max_new_tokens=150, seed=seed, constrained=True)
        programs += [p for p in text.split("\n\n") if p.strip()]
    assert programs
    assert parse_rate(programs) == 1.0
