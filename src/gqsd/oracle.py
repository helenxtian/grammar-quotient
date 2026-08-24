"""Exact p*(w) = p(w | w in L) oracle by brute-force enumeration.

Phase 1 deliverable. For a small enough L (a few hundred valid strings) we can
enumerate every valid w, score p(w) under the base model, and normalize. This is
the ground-truth distribution every correctness experiment is measured against;
without it we cannot empirically validate the correctness theorem.

Stub: the enumeration depends on the Phase-2 grammar, so this lands alongside it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .grammar import Action
from .phrase_grammar import PhraseGrammar, PhraseState

if TYPE_CHECKING:
    from .model import LM


def _logsumexp(values: list[float]) -> float:
    if not values:
        return -math.inf
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _normalize_logweights(weights: dict[str, float]) -> dict[str, float]:
    normalizer = _logsumexp(list(weights.values()))
    if not math.isfinite(normalizer):
        raise ValueError("Cannot normalize non-finite grammar mass")
    return {key: math.exp(value - normalizer) for key, value in weights.items()}


def exact_grammar_conditioned(lm: LM, valid_strings: list[str]) -> dict[str, float]:
    """Return p*(w) for each w in `valid_strings`.

    p*(w) = p(w) / sum_{w' in L} p(w'). Caller supplies the (small) enumeration
    of L; for the toy fixed-schema grammar this is a Cartesian product over the
    finite field/enum choices.
    """
    logps = {text: lm.text_logprob("", text) for text in valid_strings}
    return _normalize_logweights(logps)


@dataclass
class FiniteGrammarOracle:
    """Exact target and quotient conditionals for an enumerable phrase grammar."""

    grammar: PhraseGrammar
    sequence_logprob: Callable[[str], float]
    _completion_cache: dict[PhraseState, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        strings = self.grammar.enumerate()
        if len(strings) != len(set(strings)):
            raise ValueError("Grammar paths must produce unique terminal strings")
        self.logprobs = {text: self.sequence_logprob(text) for text in strings}
        self.p_star = _normalize_logweights(self.logprobs)

    def completion_logmass(self, state: PhraseState) -> float:
        """Unnormalized model mass of all valid terminal strings below state."""
        cached = self._completion_cache.get(state)
        if cached is not None:
            return cached
        if state.is_accepting():
            result = self.logprobs[state.text]
        else:
            result = _logsumexp(
                [
                    self.completion_logmass(state.advance(action, realization))
                    for action in state.actions()
                    for realization in action.realizations
                ]
            )
        self._completion_cache[state] = result
        return result

    def action_distribution(self, state: PhraseState) -> dict[str, float]:
        """Exact grammar-conditioned distribution over the next macro action."""
        if state.is_accepting():
            return {}
        weights = {
            action.label: _logsumexp(
                [
                    self.completion_logmass(state.advance(action, realization))
                    for realization in action.realizations
                ]
            )
            for action in state.actions()
        }
        return _normalize_logweights(weights)

    def realization_distribution(
        self, state: PhraseState, action: Action
    ) -> dict[str, float]:
        """Exact target conditional over text realizations within an action."""
        if action not in state.actions():
            raise ValueError(f"Action {action.label!r} is not available from this state")
        weights = {
            realization: self.completion_logmass(state.advance(action, realization))
            for realization in action.realizations
        }
        return _normalize_logweights(weights)

    def analytic_quotient_distribution(self) -> dict[str, float]:
        """Multiply quotient conditionals to recover the terminal distribution."""
        result: dict[str, float] = {}

        def walk(state: PhraseState, probability: float) -> None:
            if state.is_accepting():
                result[state.text] = probability
                return
            action_probs = self.action_distribution(state)
            for action in state.actions():
                realization_probs = self.realization_distribution(state, action)
                for realization, realization_prob in realization_probs.items():
                    walk(
                        state.advance(action, realization),
                        probability * action_probs[action.label] * realization_prob,
                    )

        walk(self.grammar.start(), 1.0)
        return result
