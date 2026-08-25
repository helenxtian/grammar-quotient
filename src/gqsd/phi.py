"""Approximate online estimates of grammar-conditioned future validity."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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

    def edge_logprob(self, state: PhraseState, realization: str) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class PhiComparison:
    """State-level error summary against an exact finite-grammar oracle."""

    states: int
    mean_absolute_log_ratio: float
    maximum_absolute_log_ratio: float
    support_loss: int
    exact_zero_mass: int


def finite_horizon_tv_bound(horizon: int, log_mass_error: float) -> float:
    """Bound terminal TV from multiplicative local future-mass errors.

    If every local estimate has absolute log-ratio error at most
    ``log_mass_error``, a depth-``horizon`` terminal weight has error at most
    ``horizon * log_mass_error`` before normalization.  Normalization can
    double that log-ratio, giving the sharp two-sided TV bound below.
    """
    if horizon < 0:
        raise ValueError("Horizon must be nonnegative")
    if log_mass_error < 0.0:
        raise ValueError("Log-mass error must be nonnegative")
    return math.tanh(horizon * log_mass_error)


def _normalize_logweights(weights: dict[str, float]) -> dict[str, float]:
    maximum = max(weights.values())
    total = sum(math.exp(value - maximum) for value in weights.values())
    return {key: math.exp(value - maximum) / total for key, value in weights.items()}


def approximate_quotient_distribution(
    estimator: PhiEstimator, grammar: PhraseGrammar
) -> dict[str, float]:
    """Construct the terminal distribution induced by an estimated ``Phi``."""
    if isinstance(estimator, BeamPhiEstimator) and estimator.grammar is not grammar:
        raise ValueError("Estimator and grammar must match")
    result: dict[str, float] = {}

    def walk(state: PhraseState, probability: float) -> None:
        if state.is_accepting():
            result[state.text] = probability
            return
        edges: dict[str, tuple[PhraseState, float]] = {}
        for action in state.actions():
            for realization in action.realizations:
                next_state = state.advance(action, realization)
                future = estimator.estimate(next_state).log_mass
                edge = estimator.edge_logprob(state, realization)
                edges[f"{action.label}\0{realization}"] = (next_state, edge + future)
        probabilities = _normalize_logweights(
            {key: weight for key, (_, weight) in edges.items()}
        )
        for key, (next_state, _) in edges.items():
            walk(next_state, probability * probabilities[key])

    walk(grammar.start(), 1.0)
    return result


def distribution_kl(distribution: dict[str, float], target: dict[str, float]) -> float:
    """Return ``KL(target || distribution)`` over identical finite support."""
    if set(distribution) != set(target):
        raise ValueError("Distributions must have identical support")
    return sum(
        target[key] * math.log(target[key] / distribution[key])
        for key in target
        if target[key] > 0.0
    )


def distribution_tv(distribution: dict[str, float], target: dict[str, float]) -> float:
    """Return total variation distance over identical finite support."""
    if set(distribution) != set(target):
        raise ValueError("Distributions must have identical support")
    return 0.5 * sum(abs(distribution[key] - target[key]) for key in target)


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
    score_batch_size: int = 32
    _estimate_cache: dict[PhraseState, PhiEstimate] = field(
        default_factory=dict, init=False, repr=False
    )
    _edge_cache: dict[tuple[PhraseState, str], float] = field(
        default_factory=dict, init=False, repr=False
    )
    _text_logprob_cache: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.beam_size <= 0:
            raise ValueError("Beam size must be positive")
        if self.score_batch_size <= 0:
            raise ValueError("Score batch size must be positive")

    def edge_logprob(self, state: PhraseState, realization: str) -> float:
        cached = self._edge_cache.get((state, realization))
        if cached is not None:
            return cached
        self._ensure_text_scores()
        joint = self._text_logprob_cache[state.text + realization]
        if not state.text:
            result = joint
        else:
            result = joint - self._text_logprob_cache[state.text]
        self._edge_cache[(state, realization)] = result
        return result

    def _ensure_text_scores(self) -> None:
        if self._text_logprob_cache:
            return
        texts = {""}
        for state in reachable_states(self.grammar):
            texts.add(state.text)
            if not state.is_accepting():
                texts.update(
                    state.text + realization
                    for action in state.actions()
                    for realization in action.realizations
                )
        ordered_texts = sorted(texts)
        scores = self.lm.batch_text_logprobs(
            "", ordered_texts, batch_size=self.score_batch_size
        )
        self._text_logprob_cache = dict(zip(ordered_texts, scores, strict=True))

    def estimate(self, state: PhraseState) -> PhiEstimate:
        cached = self._estimate_cache.get(state)
        if cached is not None:
            return cached
        if state.grammar is not self.grammar:
            raise ValueError("State does not belong to this estimator's grammar")
        if state.is_accepting():
            result = PhiEstimate(0.0, 1, 0, 0)
            self._estimate_cache[state] = result
            return result

        frontier: list[tuple[PhraseState, float]] = [(state, 0.0)]
        completed: list[float] = []
        expanded_paths = 0
        pruned_paths = 0

        while frontier:
            next_frontier: list[tuple[PhraseState, float]] = []
            edges: list[tuple[PhraseState, str, PhraseState, float]] = []
            for current, path_logprob in frontier:
                if current.is_accepting():
                    completed.append(path_logprob)
                    continue
                for action in current.actions():
                    for realization in action.realizations:
                        next_state = current.advance(action, realization)
                        edges.append((current, realization, next_state, path_logprob))
                        expanded_paths += 1
            if not edges:
                break
            texts = {
                text
                for current, realization, _, _ in edges
                for text in (current.text, current.text + realization)
            }
            scores = self.lm.batch_text_logprobs(
                "", texts, batch_size=self.score_batch_size
            )
            text_logprobs = dict(zip(texts, scores, strict=True))
            for current, realization, next_state, path_logprob in edges:
                logprob = text_logprobs[current.text + realization]
                if current.text:
                    logprob -= text_logprobs[current.text]
                next_frontier.append((next_state, path_logprob + logprob))
            next_frontier.sort(key=lambda item: item[1], reverse=True)
            pruned_paths += max(0, len(next_frontier) - self.beam_size)
            frontier = next_frontier[: self.beam_size]

        if not completed:
            result = PhiEstimate(-math.inf, 0, expanded_paths, pruned_paths)
        else:
            maximum = max(completed)
            log_mass = maximum + math.log(
                sum(math.exp(value - maximum) for value in completed)
            )
            result = PhiEstimate(log_mass, len(completed), expanded_paths, pruned_paths)
        self._estimate_cache[state] = result
        return result