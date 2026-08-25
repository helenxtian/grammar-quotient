"""Approximate online estimates of grammar-conditioned future validity."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .model import LM
from .oracle import FiniteGrammarOracle
from .phrase_grammar import PhraseGrammar, PhraseState


@dataclass(frozen=True)
class PhiEstimate:
    """Estimated future-validity mass and the work used to obtain it."""

    log_mass: float
    completed_paths: int
    expanded_paths: int
    pruned_paths: int

    @property
    def mass(self) -> float:
        return math.exp(self.log_mass) if math.isfinite(self.log_mass) else 0.0


class PhiEstimator:
    """Interface for estimating ``Phi(u)`` at an online grammar state."""

    def estimate(self, state: PhraseState) -> PhiEstimate:
        raise NotImplementedError


@dataclass(frozen=True)
class PhiComparison:
    """State-level error summary against an exact finite-grammar oracle."""

    states: int
    mean_absolute_log_ratio: float
    maximum_absolute_log_ratio: float
    support_loss: int
    exact_zero_mass: int


def reachable_states(grammar: PhraseGrammar) -> list[PhraseState]:
    """Return each reachable grammar state once, including the terminal state."""
    result: list[PhraseState] = []
    pending = [grammar.start()]
    seen: set[PhraseState] = set()
    while pending:
        state = pending.pop()
        if state in seen:
            continue
        seen.add(state)
        result.append(state)
        if not state.is_accepting():
            pending.extend(
                state.advance(action, realization)
                for action in state.actions()
                for realization in action.realizations
            )
    return result


def compare_phi(
    estimator: PhiEstimator,
    oracle: FiniteGrammarOracle,
    lm: LM,
) -> PhiComparison:
    """Compare an estimator with exact future mass over reachable states."""
    errors: list[float] = []
    support_loss = 0
    exact_zero_mass = 0
    for state in reachable_states(oracle.grammar):
        exact_log_mass = 0.0 if state.is_accepting() else (
            oracle.completion_logmass(state) - lm.text_logprob("", state.text)
        )
        estimate = estimator.estimate(state)
        if not math.isfinite(exact_log_mass):
            exact_zero_mass += 1
            continue
        if not math.isfinite(estimate.log_mass):
            support_loss += 1
            continue
        errors.append(abs(estimate.log_mass - exact_log_mass))
    return PhiComparison(
        states=len(reachable_states(oracle.grammar)),
        mean_absolute_log_ratio=sum(errors) / len(errors) if errors else math.inf,
        maximum_absolute_log_ratio=max(errors, default=math.inf),
        support_loss=support_loss,
        exact_zero_mass=exact_zero_mass,
    )


@dataclass
class BeamPhiEstimator(PhiEstimator):
    """Estimate future validity by retaining the highest-mass grammar paths."""

    lm: LM
    grammar: PhraseGrammar
    beam_size: int = 4

    def __post_init__(self) -> None:
        if self.beam_size <= 0:
            raise ValueError("Beam size must be positive")

    def _realization_logprob(self, prefix: str, realization: str) -> float:
        try:
            return self.lm.text_logprob(prefix, realization)
        except ValueError as error:
            if "tokenization-stable" not in str(error):
                raise
            # The online frontier owns reclamation for unstable boundaries.
            # This fallback keeps the estimator usable at an intermediate
            # grammar state; callers should supply the frontier-aware context
            # when exact boundary accounting is required.
            prefix_ids = self.lm.encode(prefix)
            return self.lm.sequence_logprob(prefix_ids, self.lm.encode(realization))

    def estimate(self, state: PhraseState) -> PhiEstimate:
        if state.grammar is not self.grammar:
            raise ValueError("State does not belong to this estimator's grammar")
        if state.is_accepting():
            return PhiEstimate(0.0, 1, 0, 0)

        frontier: list[tuple[PhraseState, float]] = [(state, 0.0)]
        completed: list[float] = []
        expanded_paths = 0
        pruned_paths = 0

        while frontier:
            next_frontier: list[tuple[PhraseState, float]] = []
            for current, path_logprob in frontier:
                if current.is_accepting():
                    completed.append(path_logprob)
                    continue
                for action in current.actions():
                    for realization in action.realizations:
                        next_state = current.advance(action, realization)
                        logprob = self._realization_logprob(current.text, realization)
                        next_frontier.append((next_state, path_logprob + logprob))
                        expanded_paths += 1
            if not next_frontier:
                break
            next_frontier.sort(key=lambda item: item[1], reverse=True)
            pruned_paths += max(0, len(next_frontier) - self.beam_size)
            frontier = next_frontier[: self.beam_size]

        if not completed:
            return PhiEstimate(-math.inf, 0, expanded_paths, pruned_paths)
        maximum = max(completed)
        log_mass = maximum + math.log(
            sum(math.exp(value - maximum) for value in completed)
        )
        return PhiEstimate(log_mass, len(completed), expanded_paths, pruned_paths)