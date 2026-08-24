"""Online local-target verification at grammar-action and token granularity."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from .grammar import Action
from .model import LM
from .phrase_grammar import PhraseGrammar, PhraseState
from .speculative import speculative_pick
from .tokenizer_analysis import TokenFrontier


@dataclass(frozen=True)
class Candidate:
    key: str
    action: Action
    realization: str
    continuation_ids: tuple[int, ...]
    verification_ids: tuple[int, ...]


@dataclass(frozen=True)
class OnlineSample:
    text: str
    token_ids: tuple[int, ...]
    target_forward_passes: int
    target_rows_scored: int
    verified_tokens: int
    accepted_tokens: int
    rejected_units: int
    reclaimed_boundaries: int
    latency_seconds: float


def _candidate_key(action: Action, realization: str) -> str:
    return f"{action.label}\0{realization}"


def _candidates(lm: LM, state: PhraseState, frontier: TokenFrontier) -> list[Candidate]:
    candidates: list[Candidate] = []
    committed = frontier.committed_ids
    for action in state.actions():
        for realization in action.realizations:
            joint_ids = tuple(lm.encode(state.text + realization))
            if joint_ids[: len(committed)] != committed:
                raise ValueError("Candidate changes tokens before the pending frontier")
            continuation_ids = joint_ids[len(committed) :]
            next_state = state.advance(action, realization)
            verification_ids = continuation_ids
            if next_state.is_accepting():
                eos_token_id = lm.tokenizer.eos_token_id
                if eos_token_id is None:
                    raise ValueError("Tokenizer needs EOS for terminal token verification")
                verification_ids += (eos_token_id,)
            candidates.append(
                Candidate(
                    key=_candidate_key(action, realization),
                    action=action,
                    realization=realization,
                    continuation_ids=continuation_ids,
                    verification_ids=verification_ids,
                )
            )
    return candidates


def _normalize_logweights(weights: dict[str, float]) -> dict[str, float]:
    maximum = max(weights.values())
    shifted = {key: math.exp(value - maximum) for key, value in weights.items()}
    total = sum(shifted.values())
    return {key: value / total for key, value in shifted.items()}


def _uniform(keys: list[str]) -> dict[str, float]:
    probability = 1.0 / len(keys)
    return {key: probability for key in keys}


def sample_online_actions(
    lm: LM,
    grammar: PhraseGrammar,
    *,
    rng: random.Random,
    pending_token_budget: int = 1,
) -> OnlineSample:
    """Verify each competing grammar-action batch with one target forward pass."""
    lm.reset_counters()
    started = time.perf_counter()
    state = grammar.start()
    frontier = TokenFrontier(lm.tokenizer, pending_token_budget=pending_token_budget)
    target_rows_scored = 0
    verified_tokens = 0
    accepted_tokens = 0
    rejected_units = 0
    reclaimed_boundaries = 0

    while not state.is_accepting():
        candidates = _candidates(lm, state, frontier)
        examples = [
            (list(frontier.committed_ids), list(candidate.verification_ids))
            for candidate in candidates
        ]
        scores = lm.batch_sequence_logprobs(examples)
        target = _normalize_logweights(
            {
                candidate.key: score
                for candidate, score in zip(candidates, scores, strict=True)
            }
        )
        chosen_key, accepted = speculative_pick(rng, target, _uniform(list(target)))
        chosen = next(candidate for candidate in candidates if candidate.key == chosen_key)
        target_rows_scored += len(candidates)
        verified_tokens += len(chosen.verification_ids)
        if accepted:
            accepted_tokens += len(chosen.verification_ids)
        else:
            rejected_units += 1
        update = frontier.append(chosen.realization)
        reclaimed_boundaries += update.reclaimed_boundary
        state = state.advance(chosen.action, chosen.realization)

    frontier.finalize()
    return OnlineSample(
        text=state.text,
        token_ids=frontier.canonical_ids,
        target_forward_passes=lm.target_forward_passes,
        target_rows_scored=target_rows_scored,
        verified_tokens=verified_tokens,
        accepted_tokens=accepted_tokens,
        rejected_units=rejected_units,
        reclaimed_boundaries=reclaimed_boundaries,
        latency_seconds=time.perf_counter() - started,
    )


def sample_online_tokens(
    lm: LM,
    grammar: PhraseGrammar,
    *,
    rng: random.Random,
    pending_token_budget: int = 1,
) -> OnlineSample:
    """Verify the same candidate set through locally masked token decisions."""
    lm.reset_counters()
    started = time.perf_counter()
    state = grammar.start()
    frontier = TokenFrontier(lm.tokenizer, pending_token_budget=pending_token_budget)
    target_rows_scored = 0
    verified_tokens = 0
    accepted_tokens = 0
    rejected_units = 0
    reclaimed_boundaries = 0

    while not state.is_accepting():
        candidates = _candidates(lm, state, frontier)
        remaining = candidates
        token_path: tuple[int, ...] = ()
        while True:
            complete = [
                candidate
                for candidate in remaining
                if len(candidate.verification_ids) == len(token_path)
            ]
            if complete:
                if len(complete) != len(remaining):
                    raise ValueError("Token baseline requires prefix-free action realizations")
                chosen = complete[0]
                break

            next_counts: dict[int, int] = {}
            for candidate in remaining:
                token_id = candidate.verification_ids[len(token_path)]
                next_counts[token_id] = next_counts.get(token_id, 0) + 1
            context_ids = list(frontier.committed_ids + token_path)
            logprobs = lm.next_token_logprobs(context_ids)
            target = _normalize_logweights(
                {str(token_id): logprobs[token_id].item() for token_id in next_counts}
            )
            draft = {
                str(token_id): count / len(remaining)
                for token_id, count in next_counts.items()
            }
            selected, accepted = speculative_pick(rng, target, draft)
            selected_id = int(selected)
            verified_tokens += 1
            accepted_tokens += accepted
            rejected_units += not accepted
            token_path += (selected_id,)
            remaining = [
                candidate
                for candidate in remaining
                if candidate.verification_ids[: len(token_path)] == token_path
            ]
            target_rows_scored += 1

        update = frontier.append(chosen.realization)
        reclaimed_boundaries += update.reclaimed_boundary
        state = state.advance(chosen.action, chosen.realization)

    frontier.finalize()
    return OnlineSample(
        text=state.text,
        token_ids=frontier.canonical_ids,
        target_forward_passes=lm.target_forward_passes,
        target_rows_scored=target_rows_scored,
        verified_tokens=verified_tokens,
        accepted_tokens=accepted_tokens,
        rejected_units=rejected_units,
        reclaimed_boundaries=reclaimed_boundaries,
        latency_seconds=time.perf_counter() - started,
    )