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

from .grammar import GrammarState
from .model import LM


def sample_unconstrained(lm: LM, prompt: str, max_tokens: int) -> str:
    """Plain ancestral sampling from p (no grammar). TODO(phase-1)."""
    raise NotImplementedError("phase 1")


def sample_token_masked(
    lm: LM, prompt: str, start_state: GrammarState, max_tokens: int
) -> str:
    """Token-level masked + renormalized decoding (the biased baseline).

    At each step: mask tokens outside the grammar's valid next-token set A(u),
    renormalize, sample. TODO(phase-1, needs phase-2 grammar).
    """
    raise NotImplementedError("phase 1")
