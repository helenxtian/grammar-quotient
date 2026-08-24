import random

import pytest

from gqsd.oracle import FiniteGrammarOracle
from gqsd.phrase_grammar import Choice, Literal, PhraseGrammar, Slot
from gqsd.quotient import sample_speculative_quotient


class MergeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"": [], "a": [1], "ab": [3], "ac": [1, 4]}[text]

    def convert_ids_to_tokens(self, ids):
        return [{1: "a", 3: "ab", 4: "c"}[token_id] for token_id in ids]


def _oracle():
    grammar = PhraseGrammar(
        name="integrated_repair",
        segments=(
            Literal("a"),
            Slot("ending", (Choice("merged", ("b",)), Choice("stable", ("c",)))),
        ),
    )
    return FiniteGrammarOracle(grammar, lambda text: {"ab": -0.1, "ac": -2.0}[text])


def test_speculative_quotient_reclaims_boundary_and_returns_canonical_ids():
    sample = sample_speculative_quotient(
        _oracle(), MergeTokenizer(), rng=random.Random(4), pending_token_budget=1
    )
    assert sample.text == "ab"
    assert sample.token_ids == (3,)
    assert sample.reclaimed_boundaries == 1
    assert sample.action_trials == 2
    assert sample.realization_trials == 2


def test_strict_token_commit_rejects_path_reclaimed_by_action_frontier():
    with pytest.raises(ValueError, match="only 0 are available"):
        sample_speculative_quotient(
            _oracle(), MergeTokenizer(), rng=random.Random(4), pending_token_budget=0
        )


def test_integrated_sampler_tracks_exact_terminal_distribution():
    oracle = _oracle()
    rng = random.Random(91)
    samples = [
        sample_speculative_quotient(
            oracle, MergeTokenizer(), rng=rng, pending_token_budget=1
        )
        for _ in range(5000)
    ]
    empirical_ab = sum(sample.text == "ab" for sample in samples) / len(samples)
    assert empirical_ab == pytest.approx(oracle.p_star["ab"], abs=0.02)
    assert all(sample.token_ids in {(3,), (1, 4)} for sample in samples)