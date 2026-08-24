"""Exact speculative correction over finite token or grammar-action supports."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping

Distribution = Mapping[str, float]


def validate_distribution(distribution: Distribution, support: set[str]) -> None:
    if set(distribution) != support:
        raise ValueError("Target and draft distributions must have identical support")
    if any(not math.isfinite(value) or value < 0.0 for value in distribution.values()):
        raise ValueError("Distribution probabilities must be finite and nonnegative")
    if not math.isclose(sum(distribution.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Distribution probabilities must sum to one")


def positive_residual(target: Distribution, draft: Distribution) -> dict[str, float]:
    """Return normalized ``(target - draft)+`` after validating both inputs."""
    support = set(target)
    validate_distribution(target, support)
    validate_distribution(draft, support)
    residual = {key: max(0.0, target[key] - draft[key]) for key in support}
    total = sum(residual.values())
    if total <= 1e-15:
        return dict(target)
    return {key: value / total for key, value in residual.items()}


def _sample(rng: random.Random, distribution: Distribution) -> str:
    threshold = rng.random()
    cumulative = 0.0
    last = ""
    for key, probability in distribution.items():
        if probability == 0.0:
            continue
        cumulative += probability
        last = key
        if threshold < cumulative:
            return key
    return last


def speculative_pick(
    rng: random.Random, target: Distribution, draft: Distribution
) -> tuple[str, bool]:
    """Sample exactly from target using one draft proposal and residual fallback."""
    support = set(target)
    validate_distribution(target, support)
    validate_distribution(draft, support)
    proposal = _sample(rng, draft)
    draft_probability = draft[proposal]
    acceptance = min(1.0, target[proposal] / draft_probability)
    if rng.random() <= acceptance:
        return proposal, True
    return _sample(rng, positive_residual(target, draft)), False


def analytic_corrected_distribution(
    target: Distribution, draft: Distribution
) -> dict[str, float]:
    """Return the exact output law of :func:`speculative_pick` for verification."""
    support = set(target)
    validate_distribution(target, support)
    validate_distribution(draft, support)
    accepted = {key: min(target[key], draft[key]) for key in support}
    rejection_probability = 1.0 - sum(accepted.values())
    residual = positive_residual(target, draft)
    return {
        key: accepted[key] + rejection_probability * residual[key]
        for key in support
    }