import numpy as np

from pyllm.checkpoint import load, save
from pyllm.generate import generate
from pyllm.models import GPT
from pyllm.tokenizer import CharTokenizer


def test_save_load_roundtrips_config_and_weights(tmp_path):
    text = "pikachu"
    tok = CharTokenizer(text)
    model = GPT(
        vocab_size=tok.vocab_size,
        block_size=8,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        rng=np.random.default_rng(0),
    )
    path = tmp_path / "model.npz"
    save(path, model, tok)
    reloaded, reloaded_tok = load(path)
    assert reloaded.config() == model.config()
    for a, b in zip(model.parameters(), reloaded.parameters(), strict=True):
        assert np.allclose(a.data, b.data)


def test_reloaded_tokenizer_matches(tmp_path):
    text = "bulbasaur"
    tok = CharTokenizer(text)
    model = GPT(
        vocab_size=tok.vocab_size,
        block_size=8,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        rng=np.random.default_rng(0),
    )
    path = tmp_path / "m.npz"
    save(path, model, tok)
    _, reloaded_tok = load(path)
    assert reloaded_tok.stoi == tok.stoi
    assert reloaded_tok.decode(reloaded_tok.encode("bulba")) == "bulba"


def test_reloaded_model_generates_identically(tmp_path):
    text = "charmander"
    tok = CharTokenizer(text)
    model = GPT(
        vocab_size=tok.vocab_size,
        block_size=8,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        rng=np.random.default_rng(0),
    )
    path = tmp_path / "m.npz"
    save(path, model, tok)
    reloaded, reloaded_tok = load(path)
    a = generate(
        model, tok, prompt="c", max_new_tokens=10, rng=np.random.default_rng(3)
    )
    b = generate(
        reloaded,
        reloaded_tok,
        prompt="c",
        max_new_tokens=10,
        rng=np.random.default_rng(3),
    )
    assert a == b
