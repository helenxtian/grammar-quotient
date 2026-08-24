import math
import random

import pytest

from gqsd.step1_toy import (
    SyntheticToyModel,
    ToyOracle,
    analytic_quotient_distribution,
    analytic_token_level_distribution,
    build_oracle,
    compile_actions,
    enumerate_all_valid_strings,
    kl_divergence,
    run_step1_exact,
    sample_quotient_level,
    sample_token_level,
)

GRANULARITIES = ("three_digits", "id_1000")


def _build_oracle(seed: int = 0) -> ToyOracle:
    valid = enumerate_all_valid_strings()
    vocab = sorted(set("".join(valid)))
    model = SyntheticToyModel(vocabulary=vocab, seed=seed)
    return ToyOracle(model=model, valid_strings=valid)


def test_enumeration_size_and_shape():
    valid = enumerate_all_valid_strings()
    assert len(valid) == 2000
    assert '{"status": "approved", "id": 000}' in valid
    assert '{"status": "denied", "id": 999}' in valid


def test_exact_p_star_normalized():
    oracle = _build_oracle(seed=3)
    total = sum(oracle.p_star.values())
    assert math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_phi_is_nonnegative_and_zero_on_invalid_prefix():
    oracle = _build_oracle(seed=7)
    assert oracle.phi("not-a-valid-prefix") == 0.0
    assert oracle.phi('{"status": ') > 0.0


def test_phi_at_root_equals_language_mass():
    # Phi("") telescopes to p(L) = sum of raw base-model masses over L.
    oracle = _build_oracle(seed=5)
    p_of_l = sum(oracle._raw_probs.values())
    assert math.isclose(oracle.phi(""), p_of_l, rel_tol=1e-12)


def test_compile_actions_supports_both_id_granularities():
    a = compile_actions("three_digits")
    b = compile_actions("id_1000")
    assert [x.label for x in a] == [
        "emit_prefix",
        "choose_status",
        "emit_id_prefix",
        "choose_id_d1",
        "choose_id_d2",
        "choose_id_d3",
        "emit_suffix",
    ]
    assert [x.label for x in b] == [
        "emit_prefix",
        "choose_status",
        "emit_id_prefix",
        "choose_id",
        "emit_suffix",
    ]


# --- Exact (noise-free) correctness of the quotient/acceptance machinery. ---


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_token_level_phi_is_exactly_p_star(granularity):
    oracle, _ = build_oracle(seed=0, id_granularity=granularity)
    dist = analytic_token_level_distribution(oracle, use_phi=True)
    for w in oracle.valid_strings:
        assert math.isclose(dist[w], oracle.p_star[w], rel_tol=0.0, abs_tol=1e-12)
    assert abs(kl_divergence(dist, oracle.p_star)) < 1e-12


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_quotient_phi_is_exactly_p_star(granularity):
    oracle, actions = build_oracle(seed=0, id_granularity=granularity)
    dist = analytic_quotient_distribution(oracle, actions, use_phi=True)
    for w in oracle.valid_strings:
        assert math.isclose(dist[w], oracle.p_star[w], rel_tol=0.0, abs_tol=1e-12)
    assert abs(kl_divergence(dist, oracle.p_star)) < 1e-12


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_quotient_phi_matches_token_phi_exactly(granularity):
    # This is the cleanest validation of the correctness theorem: changing the
    # unit of speculation must not change the sampled distribution.
    oracle, actions = build_oracle(seed=1, id_granularity=granularity)
    token = analytic_token_level_distribution(oracle, use_phi=True)
    quotient = analytic_quotient_distribution(oracle, actions, use_phi=True)
    for w in oracle.valid_strings:
        assert math.isclose(token[w], quotient[w], rel_tol=0.0, abs_tol=1e-12)


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_no_phi_conditions_measurably_fail(granularity):
    # The correctness claim is nontrivial only if the no-Phi baselines actually
    # deviate from p*. They must, and by a clearly non-negligible margin.
    kl = run_step1_exact(seed=0, id_granularity=granularity)
    assert kl["token_mask_no_phi"] > 0.05
    assert kl["quotient_no_phi"] > 0.05
    assert abs(kl["token_mask_phi"]) < 1e-12
    assert abs(kl["quotient_phi"]) < 1e-12


# --- The accept/reject rule preserves the target for any draft. ---


def test_speculative_sampler_is_target_preserving_under_uniform_draft():
    # Draft != target exercises the accept/reject + residual path. The empirical
    # distribution must still track p* (loose tolerance for finite samples).
    oracle, actions = build_oracle(seed=2, id_granularity="three_digits")
    rng = random.Random(123)
    stats = {"trials": 0, "accepts": 0}
    counts: dict[str, int] = {}
    n = 4000
    for _ in range(n):
        w = sample_quotient_level(
            oracle, actions, use_phi=True, draft_mode="uniform", rng=rng, stats=stats
        )
        counts[w] = counts.get(w, 0) + 1

    # A uniform draft genuinely rejects a lot, so the machinery is exercised.
    assert stats["trials"] > 0
    assert stats["accepts"] < stats["trials"]

    p_hat = {w: counts.get(w, 0) / n for w in oracle.valid_strings}
    kl = kl_divergence(p_hat, oracle.p_star)
    assert kl < 0.15  # finite-sample noise only; no systematic bias


def test_target_draft_accepts_everything():
    oracle, actions = build_oracle(seed=4, id_granularity="three_digits")
    rng = random.Random(7)
    stats = {"trials": 0, "accepts": 0}
    for _ in range(200):
        sample_quotient_level(
            oracle, actions, use_phi=True, draft_mode="target", rng=rng, stats=stats
        )
    assert stats["accepts"] == stats["trials"]


def test_samplers_only_produce_valid_strings():
    oracle, actions = build_oracle(seed=6, id_granularity="id_1000")
    rng = random.Random(99)
    for _ in range(100):
        w1 = sample_token_level(oracle, use_phi=True, draft_mode="target", rng=rng)
        w2 = sample_quotient_level(
            oracle, actions, use_phi=True, draft_mode="target", rng=rng
        )
        assert w1 in oracle._valid_set
        assert w2 in oracle._valid_set
