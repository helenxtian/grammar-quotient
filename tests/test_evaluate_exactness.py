from test_online import FakeModel, MergeTokenizer

from gqsd.evaluate_exactness import run_exactness_benchmark
from gqsd.model import LM
from gqsd.oracle import FiniteGrammarOracle
from gqsd.phrase_grammar import Choice, PhraseGrammar, Slot


def test_exactness_benchmark_reports_control_and_phi_rows():
    grammar = PhraseGrammar(
        name="exactness",
        segments=(Slot("value", (Choice("ab", ("ab",)), Choice("ac", ("ac",)))),),
    )
    lm = LM(tokenizer=MergeTokenizer(), model=FakeModel(), device="cpu")
    oracle = FiniteGrammarOracle.from_lm(grammar, lm)

    results = run_exactness_benchmark(
        lm, grammar, oracle, samples=2, seed=4, max_tokens=8
    )

    assert [result["decoder"] for result in results] == [
        "exact_quotient",
        "grammar_actions_no_phi",
        "grammar_actions_exact_phi",
        "grammar_actions_beam_phi",
        "token_masked",
    ]
    assert all(result["validity_rate"] == 1.0 for result in results)
    beam_result = results[3]
    assert beam_result["phi_score_batches"] >= 0
    assert beam_result["phi_scored_texts"] >= 0