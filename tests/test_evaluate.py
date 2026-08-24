import pytest

from gqsd.evaluate import run_sampling_benchmark, total_variation
from gqsd.oracle import FiniteGrammarOracle
from gqsd.phrase_grammar import Choice, Literal, PhraseGrammar, Slot


class MergeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"": [], "a": [1], "ab": [3], "ac": [1, 4]}[text]

    def convert_ids_to_tokens(self, ids):
        return [{1: "a", 3: "ab", 4: "c"}[token_id] for token_id in ids]


def _oracle():
    grammar = PhraseGrammar(
        name="evaluation_test",
        segments=(
            Literal("a"),
            Slot("ending", (Choice("merged", ("b",)), Choice("stable", ("c",)))),
        ),
    )
    return FiniteGrammarOracle(grammar, lambda text: {"ab": -0.1, "ac": -2.0}[text])


def test_sampling_benchmark_reports_reclamation_and_canonical_tokens():
    result = run_sampling_benchmark(
        _oracle(), MergeTokenizer(), samples=2000, seed=8, pending_token_budget=1
    )
    assert result["canonical_tokenizations"] == 2000
    assert result["samples_with_reclamation"] > 0
    assert result["strict_commit_rejections"] > 0
    assert result["total_variation"] < 0.03


def test_total_variation_requires_matching_support():
    with pytest.raises(ValueError, match="identical support"):
        total_variation({"a": 1.0}, {"b": 1.0})


def test_benchmark_requires_samples():
    with pytest.raises(ValueError, match="must be positive"):
        run_sampling_benchmark(_oracle(), MergeTokenizer(), samples=0, seed=1)