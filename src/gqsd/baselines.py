"""Baseline decoders to measure the quotient method against.

Phase 1 deliverable. Build these BEFORE the quotient sampler so every later
claim (faithfulness, speedup) has a reference point.

  1. unconstrained sampling from p
  2. token-level masked decoding (the standard constrained baseline -- the one
     we claim is biased relative to p* )

Stubs: token masking needs the Phase-2 grammar to define the valid next-token
set A(u), so it lands with the grammar.
"""

from __future__ import annotations

import random

from .grammar import GrammarState
from .model import LM
from .phrase_grammar import PhraseState
from .tokenizer_analysis import TokenFrontier


def sample_unconstrained(lm: LM, prompt: str, max_tokens: int) -> str:
    """Plain ancestral sampling from ``p`` until EOS or ``max_tokens``."""
    if max_tokens < 0:
        raise ValueError("Maximum token count must be nonnegative")
    rng = random.Random()
    ids = lm.encode(prompt)
    output: list[int] = []
    eos = lm.tokenizer.eos_token_id
    for _ in range(max_tokens):
        logprobs = lm.next_token_logprobs(ids)
        probabilities = logprobs.exp().tolist()
        threshold = rng.random()
        cumulative = 0.0
        selected = len(probabilities) - 1
        for token_id, probability in enumerate(probabilities):
            cumulative += probability
            if threshold < cumulative:
                selected = token_id
                break
        if eos is not None and selected == eos:
            break
        output.append(selected)
        ids.append(selected)
    return lm.decode(output)


def sample_token_masked(
    lm: LM, prompt: str, start_state: GrammarState, max_tokens: int
) -> str:
    """Token-level masked and renormalized decoding for finite phrase grammars."""
    if max_tokens < 0:
        raise ValueError("Maximum token count must be nonnegative")
    if not isinstance(start_state, PhraseState):
        raise TypeError("Token masking currently requires a PhraseState")
    rng = random.Random()
    state = start_state
    frontier = TokenFrontier(lm.tokenizer, pending_token_budget=1)
    while not state.is_accepting() and len(frontier.canonical_ids) < max_tokens:
        candidates: list[tuple[object, str, tuple[int, ...]]] = []
        for action in state.actions():
            for realization in action.realizations:
                ids = tuple(lm.encode(state.text + realization))
                if ids[: len(frontier.committed_ids)] != frontier.committed_ids:
                    continue
                candidates.append((action, realization, ids[len(frontier.committed_ids) :]))
        if not candidates:
            raise ValueError("Grammar has no tokenizable continuation")
        remaining = candidates
        path: tuple[int, ...] = ()
        while True:
            complete = [candidate for candidate in remaining if len(candidate[2]) == len(path)]
            if complete:
                chosen = complete[0]
                break
            next_ids = {candidate[2][len(path)] for candidate in remaining}
            logprobs = lm.next_token_logprobs(list(frontier.committed_ids + path))
            weights = {token_id: logprobs[token_id].exp().item() for token_id in next_ids}
            total = sum(weights.values())
            threshold = rng.random() * total
            selected = next(iter(next_ids))
            for token_id, probability in weights.items():
                threshold -= probability
                if threshold <= 0.0:
                    selected = token_id
                    break
            path += (selected,)
            remaining = [candidate for candidate in remaining if candidate[2][: len(path)] == path]
        frontier.append(chosen[1])
        state = state.advance(chosen[0], chosen[1])
    if not state.is_accepting():
        raise ValueError("Maximum token count reached before grammar acceptance")
    frontier.finalize()
    return state.text
