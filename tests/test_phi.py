import math
import random

from test_online import FakeModel, MergeTokenizer

from gqsd.model import LM
from gqsd.online import sample_online_actions
from gqsd.oracle import FiniteGrammarOracle
from gqsd.phi import BeamPhiEstimator, compare_phi
from gqsd.phrase_grammar import Choice, Literal, PhraseGrammar, Slot


class StableTokenizer(MergeTokenizer):
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"": [], "a": [1], "ab": [1, 2], "ac": [1, 4]}[text]


def _grammar():
    return PhraseGrammar(
        name="phi_test",
        segments=(
            Literal("a"),
            Slot("ending", (Choice("merged", ("b",)), Choice("stable", ("c",)))),
        ),
    )


def _lm():
    return LM(tokenizer=StableTokenizer(), model=FakeModel(), device="cpu")


def test_beam_estimator_matches_exact_future_mass_when_beam_is_complete():
    grammar = _grammar()
    lm = _lm()
    oracle = FiniteGrammarOracle.from_lm(grammar, lm)
    estimator = BeamPhiEstimator(lm, grammar, beam_size=2)

    start = grammar.start()
    state = start.advance(start.actions()[0], "a")
    estimate = estimator.estimate(state)
    expected = oracle.completion_logmass(state) - lm.text_logprob("", "a")

    assert math.isclose(estimate.log_mass, expected, rel_tol=1e-6)
    assert estimate.completed_paths == 2
    assert estimate.pruned_paths == 0


def test_small_beam_reports_pruning_and_never_returns_more_than_retained_mass():
    grammar = _grammar()
    lm = _lm()
    oracle = FiniteGrammarOracle.from_lm(grammar, lm)
    estimator = BeamPhiEstimator(lm, grammar, beam_size=1)

    start = grammar.start()
    state = start.advance(start.actions()[0], "a")
    estimate = estimator.estimate(state)
    expected = oracle.completion_logmass(state) - lm.text_logprob("", "a")

    assert estimate.pruned_paths == 1
    assert estimate.completed_paths == 1
    assert estimate.log_mass <= expected


def test_compare_phi_reports_state_level_error_metrics():
    grammar = _grammar()
    lm = _lm()
    oracle = FiniteGrammarOracle.from_lm(grammar, lm)
    comparison = compare_phi(BeamPhiEstimator(lm, grammar, beam_size=1), oracle, lm)

    assert comparison.states == 4
    assert comparison.support_loss == 0
    assert comparison.maximum_absolute_log_ratio >= 0.0


def test_online_action_sampler_accepts_phi_estimator():
    grammar = _grammar()
    sample = sample_online_actions(
        _lm(),
        grammar,
        rng=random.Random(3),
        phi_estimator=BeamPhiEstimator(_lm(), grammar, beam_size=2),
    )

    assert sample.text in {"ab", "ac"}