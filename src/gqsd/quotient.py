"""Exact finite-grammar quotient sampling with tokenizer-boundary repair."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .oracle import FiniteGrammarOracle
from .speculative import speculative_pick
from .tokenizer_analysis import TokenFrontier, Tokenizer


@dataclass(frozen=True)
class QuotientSample:
    text: str
    token_ids: tuple[int, ...]
    action_trials: int
    action_accepts: int
    realization_trials: int
    realization_accepts: int
    reclaimed_boundaries: int


def _uniform(keys: list[str]) -> dict[str, float]:
    probability = 1.0 / len(keys)
    return {key: probability for key in keys}


def sample_speculative_quotient(
    oracle: FiniteGrammarOracle,
    tokenizer: Tokenizer,
    *,
    rng: random.Random,
    pending_token_budget: int = 1,
) -> QuotientSample:
    """Sample exactly from the oracle using uniform grammar-action drafts."""
    state = oracle.grammar.start()
    frontier = TokenFrontier(tokenizer, pending_token_budget=pending_token_budget)
    action_trials = 0
    action_accepts = 0
    realization_trials = 0
    realization_accepts = 0
    reclaimed_boundaries = 0

    while not state.is_accepting():
        actions = state.actions()
        action_by_label = {action.label: action for action in actions}
        target_actions = oracle.action_distribution(state)
        chosen_label, accepted = speculative_pick(
            rng, target_actions, _uniform(list(target_actions))
        )
        action_trials += 1
        action_accepts += accepted
        action = action_by_label[chosen_label]

        target_realizations = oracle.realization_distribution(state, action)
        realization, accepted = speculative_pick(
            rng, target_realizations, _uniform(list(target_realizations))
        )
        realization_trials += 1
        realization_accepts += accepted

        update = frontier.append(realization)
        reclaimed_boundaries += update.reclaimed_boundary
        state = state.advance(action, realization)

    frontier.finalize()
    return QuotientSample(
        text=state.text,
        token_ids=frontier.canonical_ids,
        action_trials=action_trials,
        action_accepts=action_accepts,
        realization_trials=realization_trials,
        realization_accepts=realization_accepts,
        reclaimed_boundaries=reclaimed_boundaries,
    )