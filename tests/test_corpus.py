from pyllm.data import load_corpus


def test_pokemon_corpus_loads_nonempty_text():
    text = load_corpus("pokemon")
    assert isinstance(text, str)
    assert len(text) > 500


def test_pokemon_corpus_has_many_names():
    names = [line for line in load_corpus("pokemon").splitlines() if line]
    assert len(names) >= 100


def test_pokemon_corpus_is_lowercase_letters_and_newlines():
    text = load_corpus("pokemon")
    assert set(text) <= set("abcdefghijklmnopqrstuvwxyz\n .-'")
