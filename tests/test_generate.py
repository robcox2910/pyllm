import numpy as np

from pyllm.generate import generate
from pyllm.generate import sample_next
from pyllm.models import GPT
from pyllm.models import MLP
from pyllm.tokenizer import CharTokenizer


def test_greedy_picks_argmax():
    logits = np.array([0.1, 5.0, -2.0, 0.3])
    assert sample_next(logits, temperature=0.0) == 1


def test_top_k_1_is_effectively_greedy():
    logits = np.array([0.1, 5.0, -2.0, 0.3])
    rng = np.random.default_rng(0)
    picks = {sample_next(logits, temperature=1.0, top_k=1, rng=rng)
             for _ in range(20)}
    assert picks == {1}  # only the single top token is ever allowed


def test_low_temperature_concentrates_on_top_token():
    logits = np.array([0.0, 2.0, 0.0])
    rng = np.random.default_rng(0)
    picks = [sample_next(logits, temperature=0.1, rng=rng) for _ in range(200)]
    assert picks.count(1) > 190  # nearly always the peak


def test_sampling_stays_in_vocab_range():
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rng = np.random.default_rng(1)
    for _ in range(50):
        idx = sample_next(logits, temperature=1.0, rng=rng)
        assert 0 <= idx < 5


def test_generate_returns_prompt_plus_new_tokens():
    text = "abcde"
    tok = CharTokenizer(text)
    model = GPT(vocab_size=tok.vocab_size, block_size=8, embed_dim=8,
                num_heads=2, num_layers=1, rng=np.random.default_rng(0))
    out = generate(model, tok, prompt="a", max_new_tokens=5,
                   rng=np.random.default_rng(0))
    assert out.startswith("a")
    assert len(out) == 1 + 5  # prompt char + 5 generated


def test_generate_only_emits_known_characters():
    text = "pokemon"
    tok = CharTokenizer(text)
    model = GPT(vocab_size=tok.vocab_size, block_size=8, embed_dim=8,
                num_heads=2, num_layers=1, rng=np.random.default_rng(0))
    out = generate(model, tok, prompt="p", max_new_tokens=20,
                   rng=np.random.default_rng(1))
    assert set(out).issubset(set(text))


def test_generate_works_for_single_mode_model_with_short_prompt():
    text = "abcdef"
    tok = CharTokenizer(text)
    model = MLP(vocab_size=tok.vocab_size, block_size=3, embed_dim=8,
                hidden_dim=16, rng=np.random.default_rng(0))
    # empty prompt must still work (left-pad the window)
    out = generate(model, tok, prompt="", max_new_tokens=4,
                   rng=np.random.default_rng(0))
    assert len(out) == 4
    assert set(out).issubset(set(text))
