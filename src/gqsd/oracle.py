"""Exact p*(w) = p(w | w in L) oracle by brute-force enumeration.

Phase 1 deliverable. For a small enough L (a few hundred valid strings) we can
enumerate every valid w, score p(w) under the base model, and normalize. This is
the ground-truth distribution every correctness experiment is measured against;
without it we cannot empirically validate the correctness theorem.

Stub: the enumeration depends on the Phase-2 grammar, so this lands alongside it.
"""

from __future__ import annotations

from .model import LM


def exact_grammar_conditioned(lm: LM, valid_strings: list[str]) -> dict[str, float]:
    """Return p*(w) for each w in `valid_strings`.

    p*(w) = p(w) / sum_{w' in L} p(w'). Caller supplies the (small) enumeration
    of L; for the toy fixed-schema grammar this is a Cartesian product over the
    finite field/enum choices.
    """
    prefix = lm.encode("")  # BOS handling lives in the tokenizer
    logps = {}
    for w in valid_strings:
        logps[w] = lm.sequence_logprob(prefix, lm.encode(w))
    # normalize in a numerically stable way
    import math

    m = max(logps.values())
    z = sum(math.exp(lp - m) for lp in logps.values())
    return {w: math.exp(lp - m) / z for w, lp in logps.items()}
