import math
import random

from test_online import FakeModel, MergeTokenizer

from gqsd.evaluate_phi import run_phi_benchmark
from gqsd.model import LM
from gqsd.online import sample_online_actions
from gqsd.oracle import FiniteGrammarOracle
from gqsd.phi import (
    BeamPhiEstimator,
    approximate_quotient_distribution,
    compare_phi,
    distribution_kl,
    distribution_tv,
    finite_horizon_tv_bound,
)
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


def test_beam_estimator_uses_canonical_mass_at_merge_boundary():
    grammar = _grammar()
    lm = LM(tokenizer=MergeTokenizer(), model=FakeModel(), device="cpu")
    estimator = BeamPhiEstimator(lm, grammar, beam_size=2)

    start = grammar.start()
    state = start.advance(start.actions()[0], "a")
    estimate = estimator.estimate(state)
    expected = math.log(math.exp(lm.text_logprob("", "ab")) + math.exp(lm.text_logprob("", "ac")))

    assert math.isclose(estimate.log_mass, expected - lm.text_logprob("", "a"), rel_tol=1e-6)


def test_beam_estimator_batches_each_expansion_layer():
    grammar = _grammar()
    lm = _lm()
    start = grammar.start()
    state = start.advance(start.actions()[0], "a")

    BeamPhiEstimator(lm, grammar, beam_size=2).estimate(state)

    assert lm.target_forward_passes == 1


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


def test_finite_horizon_tv_bound_is_zero_without_local_error():
    assert finite_horizon_tv_bound(5, 0.0) == 0.0


def test_finite_horizon_tv_bound_grows_with_depth_and_error():
    shallow = finite_horizon_tv_bound(2, 0.1)
    deep = finite_horizon_tv_bound(4, 0.1)

    assert 0.0 < shallow < deep < 1.0


def test_complete_beam_recovers_exact_terminal_distribution():
    grammar = _grammar()
    lm = _lm()
    oracle = FiniteGrammarOracle.from_lm(grammar, lm)
    distribution = approximate_quotient_distribution(
        BeamPhiEstimator(lm, grammar, beam_size=2), grammar
    )

    assert distribution_tv(distribution, oracle.p_star) < 1e-6
    assert distribution_kl(distribution, oracle.p_star) < 1e-6


def test_phi_benchmark_reports_state_and_terminal_metrics():
    result = run_phi_benchmark(_grammar(), _lm(), beam_sizes=[1, 2], batch_size=2)

    assert result["support_size"] == 2
    assert [row["beam_size"] for row in result["results"]] == [1, 2]
    assert all("terminal_tv" in row for row in result["results"])