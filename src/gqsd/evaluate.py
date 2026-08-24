"""Run exact finite-grammar quotient benchmarks against a local causal LM."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .model import DEFAULT_MODEL, LM
from .oracle import FiniteGrammarOracle
from .phrase_grammar import PhraseGrammar
from .quotient import sample_speculative_quotient
from .tokenizer_analysis import Tokenizer


def total_variation(empirical: dict[str, float], target: dict[str, float]) -> float:
    if set(empirical) != set(target):
        raise ValueError("Empirical and target distributions must have identical support")
    return 0.5 * sum(abs(empirical[key] - target[key]) for key in target)


def run_sampling_benchmark(
    oracle: FiniteGrammarOracle,
    tokenizer: Tokenizer,
    *,
    samples: int,
    seed: int,
    pending_token_budget: int = 1,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("Sample count must be positive")
    rng = random.Random(seed)
    outputs = [
        sample_speculative_quotient(
            oracle,
            tokenizer,
            rng=rng,
            pending_token_budget=pending_token_budget,
        )
        for _ in range(samples)
    ]
    counts = Counter(output.text for output in outputs)
    empirical = {text: counts[text] / samples for text in oracle.p_star}

    strict_rejections = 0
    strict_rng = random.Random(seed)
    for _ in range(samples):
        try:
            sample_speculative_quotient(
                oracle, tokenizer, rng=strict_rng, pending_token_budget=0
            )
        except ValueError as error:
            if "pending tokens" not in str(error):
                raise
            strict_rejections += 1

    action_trials = sum(output.action_trials for output in outputs)
    realization_trials = sum(output.realization_trials for output in outputs)
    canonical = sum(
        tuple(tokenizer.encode(output.text, add_special_tokens=False)) == output.token_ids
        for output in outputs
    )
    return {
        "samples": samples,
        "seed": seed,
        "total_variation": total_variation(empirical, oracle.p_star),
        "canonical_tokenizations": canonical,
        "samples_with_reclamation": sum(
            output.reclaimed_boundaries > 0 for output in outputs
        ),
        "reclaimed_boundaries": sum(output.reclaimed_boundaries for output in outputs),
        "strict_commit_rejections": strict_rejections,
        "action_acceptance_rate": (
            sum(output.action_accepts for output in outputs) / action_trials
        ),
        "realization_acceptance_rate": (
            sum(output.realization_accepts for output in outputs) / realization_trials
        ),
        "example": outputs[0].text,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grammar", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--pending-token-budget", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    grammar = PhraseGrammar.load(args.grammar)
    lm = LM.load(
        args.model,
        device=args.device,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    started = time.perf_counter()
    oracle = FiniteGrammarOracle.from_lm(grammar, lm, batch_size=args.batch_size)
    score_seconds = time.perf_counter() - started
    result = {
        "model": args.model,
        "revision": args.revision,
        "grammar": args.grammar.name,
        "support_size": len(oracle.p_star),
        "score_seconds": score_seconds,
        "p_star_sum": math.fsum(oracle.p_star.values()),
        "pending_token_budget": args.pending_token_budget,
        **run_sampling_benchmark(
            oracle,
            lm.tokenizer,
            samples=args.samples,
            seed=args.seed,
            pending_token_budget=args.pending_token_budget,
        ),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())