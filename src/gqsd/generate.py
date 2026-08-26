"""Reusable grammar-action generation engine."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from .grammar import Action
from .model import LM
from .phi import PhiEstimator
from .phrase_grammar import PhraseGrammar
from .speculative import speculative_pick
from .tokenizer_analysis import TokenFrontier


@dataclass(frozen=True)
class ActionStep:
    label: str
    realization: str
    accepted: bool
    reclaimed_boundary: bool


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: tuple[int, ...]
    actions: tuple[ActionStep, ...]
    target_forward_passes: int
    target_rows_scored: int
    accepted_tokens: int
    rejected_actions: int
    reclaimed_boundaries: int
    latency_seconds: float


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    maximum = max(weights.values())
    values = {key: math.exp(value - maximum) for key, value in weights.items()}
    total = sum(values.values())
    return {key: value / total for key, value in values.items()}


def _uniform(keys: list[str]) -> dict[str, float]:
    probability = 1.0 / len(keys)
    return {key: probability for key in keys}


def _key(action: Action, realization: str) -> str:
    return f"{action.label}\0{realization}"


def generate_actions(
    lm: LM,
    grammar: PhraseGrammar,
    *,
    rng: random.Random | None = None,
    pending_token_budget: int = 1,
    phi_estimator: PhiEstimator | None = None,
) -> GenerationResult:
    """Generate a valid output by proposing and verifying grammar actions."""
    rng = rng or random.Random()
    lm.reset_counters()
    started = time.perf_counter()
    state = grammar.start()
    frontier = TokenFrontier(lm.tokenizer, pending_token_budget=pending_token_budget)
    actions: list[ActionStep] = []
    target_rows_scored = 0
    accepted_tokens = 0
    rejected_actions = 0
    reclaimed_boundaries = 0

    while not state.is_accepting():
        candidates: list[tuple[Action, str, tuple[int, ...], tuple[int, ...]]] = []
        for action in state.actions():
            for realization in action.realizations:
                joint_ids = tuple(lm.encode(state.text + realization))
                if joint_ids[: len(frontier.committed_ids)] != frontier.committed_ids:
                    raise ValueError("Candidate changes tokens before the pending frontier")
                continuation = joint_ids[len(frontier.committed_ids) :]
                verification = continuation
                next_state = state.advance(action, realization)
                if next_state.is_accepting():
                    if lm.tokenizer.eos_token_id is None:
                        raise ValueError("Tokenizer needs EOS for terminal verification")
                    verification += (lm.tokenizer.eos_token_id,)
                candidates.append((action, realization, continuation, verification))
        scores = lm.batch_sequence_logprobs(
            [([*frontier.committed_ids], list(verification)) for _, _, _, verification in candidates]
        )
        weights: dict[str, float] = {}
        for (action, realization, _, _), score in zip(candidates, scores, strict=True):
            next_state = state.advance(action, realization)
            future = 0.0
            if phi_estimator is not None and not next_state.is_accepting():
                future = phi_estimator.estimate(next_state).log_mass
            weights[_key(action, realization)] = score + future
        target = _normalize(weights)
        chosen_key, accepted = speculative_pick(rng, target, _uniform(list(target)))
        chosen = next(candidate for candidate in candidates if _key(candidate[0], candidate[1]) == chosen_key)
        update = frontier.append(chosen[1])
        actions.append(ActionStep(chosen[0].label, chosen[1], accepted, update.reclaimed_boundary))
        accepted_tokens += len(chosen[3]) if accepted else 0
        rejected_actions += not accepted
        reclaimed_boundaries += update.reclaimed_boundary
        target_rows_scored += len(candidates)
        state = state.advance(chosen[0], chosen[1])

    frontier.finalize()
    return GenerationResult(
        text=state.text,
        token_ids=frontier.canonical_ids,
        actions=tuple(actions),
        target_forward_passes=lm.target_forward_passes,
        target_rows_scored=target_rows_scored,
        accepted_tokens=accepted_tokens,
        rejected_actions=rejected_actions,
        reclaimed_boundaries=reclaimed_boundaries,
        latency_seconds=time.perf_counter() - started,
    )