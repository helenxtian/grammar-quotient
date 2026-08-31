"""Reusable grammar-action generation engine."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from .grammar import ActionKind
from .model import LM
from .online import _candidates
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


def _sample_open_span(
    lm: LM,
    prompt: str,
    state_text: str,
    stop: str,
    max_tokens: int,
    rng: random.Random,
) -> tuple[str, int]:
    generated = ""
    for token_count in range(1, max_tokens + 1):
        context_ids = lm.encode(prompt + state_text + generated)
        logprobs = lm.next_token_logprobs(context_ids)
        probabilities = logprobs.exp().tolist()
        threshold = rng.random()
        selected = len(probabilities) - 1
        for token_id, probability in enumerate(probabilities):
            threshold -= probability
            if threshold <= 0.0:
                selected = token_id
                break
        if lm.tokenizer.eos_token_id is not None and selected == lm.tokenizer.eos_token_id:
            raise ValueError("Open span reached EOS before its stop marker")
        generated += lm.decode([selected])
        if generated.endswith(stop):
            return generated[: -len(stop)], token_count
    raise ValueError("Open span exceeded its token budget")


def generate_actions(
    lm: LM,
    grammar: PhraseGrammar,
    *,
    prompt: str = "",
    rng: random.Random | None = None,
    pending_token_budget: int = 1,
    phi_estimator: PhiEstimator | None = None,
    use_prefix_cache: bool = True,
) -> GenerationResult:
    """Generate a valid output by proposing and verifying grammar actions."""
    rng = rng or random.Random()
    lm.reset_counters()
    started = time.perf_counter()
    state = grammar.start()
    frontier = TokenFrontier(lm.tokenizer, pending_token_budget=pending_token_budget)
    if prompt:
        frontier.append(prompt)
    actions: list[ActionStep] = []
    target_rows_scored = 0
    accepted_tokens = 0
    rejected_actions = 0
    reclaimed_boundaries = 0

    while not state.is_accepting():
        open_actions = [action for action in state.actions() if action.kind is ActionKind.OPEN]
        if open_actions:
            if len(open_actions) != 1:
                raise ValueError("Open-span decoding requires one available open action")
            segment = grammar.segments[state.segment_index]
            if not hasattr(segment, "stop") or not hasattr(segment, "max_tokens"):
                raise ValueError("OPEN action is not backed by a bounded open span")
            realization, token_count = _sample_open_span(
                lm, prompt, state.text, segment.stop, segment.max_tokens, rng
            )
            update = frontier.append(realization + segment.stop)
            actions.append(
                ActionStep(
                    open_actions[0].label,
                    realization,
                    True,
                    update.reclaimed_boundary,
                )
            )
            accepted_tokens += token_count
            reclaimed_boundaries += update.reclaimed_boundary
            state = state.advance(open_actions[0], realization + segment.stop)
            continue
        candidates = _candidates(lm, state, frontier, text_prefix=prompt)
        examples = [
            ([*frontier.committed_ids], list(candidate.verification_ids))
            for candidate in candidates
        ]
        scores = None
        if use_prefix_cache:
            scores = lm.batch_sequence_logprobs_with_prefix_cache(
                examples[0][0], [verification for _, verification in examples]
            )
        if scores is None:
            scores = lm.batch_sequence_logprobs(examples)
        weights: dict[str, float] = {}
        for candidate, score in zip(candidates, scores, strict=True):
            next_state = state.advance(candidate.action, candidate.realization)
            future = 0.0
            if phi_estimator is not None and not next_state.is_accepting():
                future = phi_estimator.estimate(next_state).log_mass
            weights[candidate.key] = score + future
        target = _normalize(weights)
        chosen_key, accepted = speculative_pick(rng, target, _uniform(list(target)))
        chosen = next(candidate for candidate in candidates if candidate.key == chosen_key)
        update = frontier.append(chosen.realization)
        actions.append(ActionStep(chosen.action.label, chosen.realization, accepted, update.reclaimed_boundary))
        accepted_tokens += len(chosen.verification_ids) if accepted else 0
        rejected_actions += not accepted
        reclaimed_boundaries += update.reclaimed_boundary
        target_rows_scored += len(candidates)
        state = state.advance(chosen.action, chosen.realization)

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