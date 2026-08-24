import math

import pytest

from gqsd.speculative import (
    analytic_corrected_distribution,
    positive_residual,
    speculative_pick,
)


@pytest.mark.parametrize(
    "draft",
    [
        {"report": 1 / 3, "dialogue": 1 / 3, "code": 1 / 3},
        {"report": 0.9, "dialogue": 0.1, "code": 0.0},
        {"report": 0.15, "dialogue": 0.25, "code": 0.6},
    ],
)
def test_residual_correction_preserves_exact_target_for_approximate_draft(draft):
    target = {"report": 0.15, "dialogue": 0.25, "code": 0.6}
    corrected = analytic_corrected_distribution(target, draft)
    for key, probability in target.items():
        assert math.isclose(corrected[key], probability, rel_tol=0.0, abs_tol=1e-12)


def test_positive_residual_reallocates_only_underweighted_outcomes():
    target = {"a": 0.2, "b": 0.3, "c": 0.5}
    draft = {"a": 0.6, "b": 0.1, "c": 0.3}
    assert positive_residual(target, draft) == pytest.approx({"a": 0.0, "b": 0.5, "c": 0.5})


def test_distribution_support_must_match():
    with pytest.raises(ValueError, match="identical support"):
        analytic_corrected_distribution({"a": 1.0}, {"b": 1.0})


def test_zero_rng_draw_does_not_select_zero_probability_proposal():
    class ZeroRng:
        def random(self):
            return 0.0

    outcome, accepted = speculative_pick(
        ZeroRng(),
        {"missing_from_draft": 0.5, "proposed": 0.5},
        {"missing_from_draft": 0.0, "proposed": 1.0},
    )
    assert outcome == "proposed"
    assert accepted is True