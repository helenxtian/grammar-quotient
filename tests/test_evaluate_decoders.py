from test_online import FakeModel, MergeTokenizer

from gqsd.evaluate_decoders import run_decoder_benchmark
from gqsd.model import LM
from gqsd.oracle import FiniteGrammarOracle
from gqsd.phrase_grammar import Choice, Literal, PhraseGrammar, Slot


def test_decoder_benchmark_reports_all_methods_and_validity():
    grammar = PhraseGrammar(
        name="benchmark",
        segments=(
            Literal("a"),
            Slot("ending", (Choice("merged", ("b",)), Choice("stable", ("c",)))),
        ),
    )
    lm = LM(tokenizer=MergeTokenizer(), model=FakeModel(), device="cpu")
    oracle = FiniteGrammarOracle.from_lm(grammar, lm)

    results = run_decoder_benchmark(
        lm, grammar, oracle, samples=3, seed=7, max_tokens=4
    )

    assert [result["decoder"] for result in results] == [
        "unconstrained",
        "token_masked",
        "grammar_actions_padded",
        "grammar_actions_cache",
        "grammar_actions_adaptive",
    ]
    assert all(result["samples"] == 3 for result in results)
    assert results[1]["validity_rate"] == 1.0
    assert all(result["validity_rate"] == 1.0 for result in results[2:])
    assert all(result["target_forward_passes"] > 0 for result in results)