import math

import pytest

from gqsd.oracle import FiniteGrammarOracle
from gqsd.phrase_grammar import Choice, Literal, PhraseGrammar, Slot


def _grammar():
    return PhraseGrammar(
        name="oracle_test",
        segments=(
            Literal("Result: "),
            Slot(
                "outcome",
                (
                    Choice("success", ("passed", "succeeded")),
                    Choice("failure", ("failed",)),
                ),
            ),
            Slot(
                "detail",
                (
                    Choice("short", (".",)),
                    Choice("long", (" after retry.", " after validation.")),
                ),
            ),
        ),
    )


def _logprob(text):
    return -0.07 * len(text) - 0.11 * text.count("a")


def test_exact_quotient_factorization_recovers_p_star():
    oracle = FiniteGrammarOracle(_grammar(), _logprob)
    quotient = oracle.analytic_quotient_distribution()
    assert quotient.keys() == oracle.p_star.keys()
    for text, probability in oracle.p_star.items():
        assert math.isclose(quotient[text], probability, rel_tol=0.0, abs_tol=1e-12)


def test_action_and_realization_distributions_are_nontrivial_and_normalized():
    oracle = FiniteGrammarOracle(_grammar(), _logprob)
    state = oracle.grammar.start()
    literal = state.actions()[0]
    state = state.advance(literal, literal.realizations[0])

    actions = oracle.action_distribution(state)
    assert set(actions) == {"outcome:success", "outcome:failure"}
    assert math.isclose(sum(actions.values()), 1.0, abs_tol=1e-12)

    success = state.actions()[0]
    realizations = oracle.realization_distribution(state, success)
    assert set(realizations) == {"passed", "succeeded"}
    assert math.isclose(sum(realizations.values()), 1.0, abs_tol=1e-12)


def test_oracle_accepts_complete_precomputed_logprobs():
    grammar = _grammar()
    logprobs = {text: _logprob(text) for text in grammar.enumerate()}
    oracle = FiniteGrammarOracle(grammar=grammar, logprobs=logprobs)
    assert oracle.logprobs == logprobs


def test_oracle_rejects_incomplete_precomputed_logprobs():
    with pytest.raises(ValueError, match="must cover"):
        FiniteGrammarOracle(grammar=_grammar(), logprobs={"missing": -1.0})