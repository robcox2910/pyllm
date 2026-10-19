"""Tests for generate."""

import numpy as np
import pytest

from pyllm.generate import generate, sample_next
from pyllm.models import GPT, MLP
from pyllm.tokenizer import CharTokenizer


def test_greedy_picks_argmax():
    """Greedy picks argmax."""
    logits = np.array([0.1, 5.0, -2.0, 0.3])
    assert sample_next(logits, temperature=0.0) == 1


def test_top_k_1_is_effectively_greedy():
    """Top k 1 is effectively greedy."""
    logits = np.array([0.1, 5.0, -2.0, 0.3])
    rng = np.random.default_rng(0)
    picks = {sample_next(logits, temperature=1.0, top_k=1, rng=rng) for _ in range(20)}
    assert picks == {1}  # only the single top token is ever allowed


def test_top_k_zero_keeps_all_tokens():
    # top_k=0 must mean "no shortlist", not "keep zero tokens".
    """Top k zero keeps all tokens."""
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rng = np.random.default_rng(0)
    picks = {sample_next(logits, temperature=1.0, top_k=0, rng=rng) for _ in range(200)}
    # every token should be reachable, so we see more than just the single best
    assert len(picks) > 1
    assert all(0 <= p < 5 for p in picks)


def test_negative_top_k_keeps_all_tokens():
    """Negative top k keeps all tokens."""
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rng = np.random.default_rng(0)
    picks = {sample_next(logits, temperature=1.0, top_k=-3, rng=rng) for _ in range(200)}
    assert len(picks) > 1
    assert all(0 <= p < 5 for p in picks)


def test_low_temperature_concentrates_on_top_token():
    """Low temperature concentrates on top token."""
    logits = np.array([0.0, 2.0, 0.0])
    rng = np.random.default_rng(0)
    picks = [sample_next(logits, temperature=0.1, rng=rng) for _ in range(200)]
    assert picks.count(1) > 190  # nearly always the peak


def test_sampling_stays_in_vocab_range():
    """Sampling stays in vocab range."""
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rng = np.random.default_rng(1)
    for _ in range(50):
        idx = sample_next(logits, temperature=1.0, rng=rng)
        assert 0 <= idx < 5


def test_generate_returns_prompt_plus_new_tokens():
    """Generate returns prompt plus new tokens."""
    text = "abcde"
    tok = CharTokenizer(text)
    model = GPT(
        vocab_size=tok.vocab_size,
        block_size=8,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        rng=np.random.default_rng(0),
    )
    out = generate(model, tok, prompt="a", max_new_tokens=5, rng=np.random.default_rng(0))
    assert out.startswith("a")
    assert len(out) == 1 + 5  # prompt char + 5 generated


def test_generate_only_emits_known_characters():
    """Generate only emits known characters."""
    text = "pokemon"
    tok = CharTokenizer(text)
    model = GPT(
        vocab_size=tok.vocab_size,
        block_size=8,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        rng=np.random.default_rng(0),
    )
    out = generate(model, tok, prompt="p", max_new_tokens=20, rng=np.random.default_rng(1))
    assert set(out).issubset(set(text))


def test_generate_sequence_model_with_empty_prompt():
    # A GPT with an empty prompt must seed itself, not crash on an empty window.
    """Generate sequence model with empty prompt."""
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
    out = generate(model, tok, prompt="", max_new_tokens=6, rng=np.random.default_rng(0))
    assert len(out) == 6
    assert set(out).issubset(set(text))


def test_generate_works_for_single_mode_model_with_short_prompt():
    """Generate works for single mode model with short prompt."""
    text = "abcdef"
    tok = CharTokenizer(text)
    model = MLP(
        vocab_size=tok.vocab_size,
        block_size=3,
        embed_dim=8,
        hidden_dim=16,
        rng=np.random.default_rng(0),
    )
    # empty prompt must still work (left-pad the window)
    out = generate(model, tok, prompt="", max_new_tokens=4, rng=np.random.default_rng(0))
    assert len(out) == 4
    assert set(out).issubset(set(text))


def test_forbidden_tokens_are_never_chosen():
    """Forbidden tokens are never chosen."""
    logits = np.array([0.0, 9.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    picks = {sample_next(logits, rng=rng, allowed=lambda token: token != 1) for _ in range(200)}
    assert 1 not in picks


def test_greedy_picks_the_best_allowed_token():
    """Greedy picks the best allowed token."""
    logits = np.array([0.1, 5.0, -2.0, 3.0])
    pick = sample_next(logits, temperature=0.0, allowed=lambda token: token != 1)
    assert pick == 3


def test_top_k_falls_back_when_the_shortlist_is_all_forbidden():
    """Top k falls back when the shortlist is all forbidden."""
    logits = np.array([0.1, 5.0, -2.0, 3.0])
    rng = np.random.default_rng(0)
    pick = sample_next(logits, top_k=1, rng=rng, allowed=lambda token: token != 1)
    assert pick != 1


def test_nothing_allowed_is_an_error():
    """Nothing allowed is an error."""
    with pytest.raises(RuntimeError):
        sample_next(np.array([1.0, 2.0]), allowed=lambda _token: False)


class OnlyAs:
    """A toy referee: text may only contain 'a', and is finished at even lengths."""

    def viable(self, text: str) -> bool:
        """Viable."""
        return set(text) <= {"a"}

    def complete(self, text: str) -> bool:
        """Complete."""
        return self.viable(text) and len(text) % 2 == 0


def _tiny_gpt(text: str = "abc") -> tuple[GPT, CharTokenizer]:
    """Build a tiny GPT and the tokenizer for `text`."""
    tokenizer = CharTokenizer(text)
    model = GPT(
        tokenizer.vocab_size,
        block_size=8,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        rng=np.random.default_rng(0),
    )
    return model, tokenizer


def test_referee_keeps_generation_legal_and_trims_to_a_finished_point():
    """Referee keeps generation legal and trims to a finished point."""
    model, tokenizer = _tiny_gpt()
    text = generate(
        model,
        tokenizer,
        max_new_tokens=7,
        rng=np.random.default_rng(0),
        referee=OnlyAs(),
    )
    assert text == "aaaaaa"  # 7 legal letters, trimmed back to the last even length


def test_greedy_generation_with_a_referee_stays_legal():
    """Greedy generation with a referee stays legal."""
    model, tokenizer = _tiny_gpt()
    text = generate(model, tokenizer, max_new_tokens=5, temperature=0.0, referee=OnlyAs())
    assert text == "aaaa"


def test_rule_breaking_prompt_is_rejected():
    """Rule breaking prompt is rejected."""
    model, tokenizer = _tiny_gpt()
    with pytest.raises(ValueError, match="breaks the rules"):
        generate(model, tokenizer, prompt="b", referee=OnlyAs())


def test_unfinished_prompt_with_no_finished_point_returns_nothing():
    """Unfinished prompt with no finished point returns nothing."""

    class NeverDone(OnlyAs):
        """Never done tests."""

        def complete(self, text: str) -> bool:  # noqa: ARG002
            """Never finish."""
            return False

    model, tokenizer = _tiny_gpt()
    text = generate(model, tokenizer, prompt="a", max_new_tokens=4, referee=NeverDone())
    assert text == ""  # never hand back half a program, not even the prompt


class OnlyAsInSixes(OnlyAs):
    """A toy referee whose programs are finished only at multiples of six letters."""

    def complete(self, text: str) -> bool:
        """Complete."""
        return self.viable(text) and len(text) % 6 == 0


def test_budget_running_out_mid_program_plays_overtime_instead_of_returning_the_prompt():
    """An already-finished prompt is not an answer: keep going until something new finishes."""
    model, tokenizer = _tiny_gpt()
    # The empty prompt counts as finished, and 4 letters is not enough to reach 6.
    text = generate(
        model,
        tokenizer,
        max_new_tokens=4,
        rng=np.random.default_rng(0),
        referee=OnlyAsInSixes(),
    )
    assert text == "aaaaaa"


def test_finishing_within_budget_does_not_play_overtime():
    """When something new finished inside the budget, the budget is still the cap."""
    model, tokenizer = _tiny_gpt()
    text = generate(
        model,
        tokenizer,
        max_new_tokens=8,
        rng=np.random.default_rng(0),
        referee=OnlyAsInSixes(),
    )
    assert text == "aaaaaa"
