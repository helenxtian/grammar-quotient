"""Compare online action-batch and token-trie target verification."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .model import DEFAULT_MODEL, LM
from .online import OnlineSample, sample_online_actions, sample_online_tokens
from .phrase_grammar import PhraseGrammar


def _summarize(
    lm: LM,
    outputs: list[OnlineSample],
    *,
    target_definition: str,
) -> dict[str, Any]:
    samples = len(outputs)
    target_calls = sum(output.target_forward_passes for output in outputs)
    accepted_tokens = sum(output.accepted_tokens for output in outputs)
    output_tokens = sum(len(output.token_ids) for output in outputs)
    latency = sum(output.latency_seconds for output in outputs)
    return {
        "target_definition": target_definition,
        "samples": samples,
        "valid_outputs": samples,
        "canonical_tokenizations": sum(
            tuple(lm.encode(output.text)) == output.token_ids for output in outputs
        ),
        "target_forward_passes": target_calls,
        "target_rows_scored": sum(output.target_rows_scored for output in outputs),
        "verified_tokens": sum(output.verified_tokens for output in outputs),
        "accepted_tokens": accepted_tokens,
        "output_tokens": output_tokens,
        "rejected_units": sum(output.rejected_units for output in outputs),
        "reclaimed_boundaries": sum(
            output.reclaimed_boundaries for output in outputs
        ),
        "latency_seconds": latency,
        "target_calls_per_sample": target_calls / samples,
        "accepted_tokens_per_target_call": accepted_tokens / target_calls,
        "output_tokens_per_target_call": output_tokens / target_calls,
        "latency_seconds_per_sample": latency / samples,
    }


def run_online_comparison(
    lm: LM,
    grammar: PhraseGrammar,
    *,
    samples: int,
    seed: int,
    pending_token_budget: int = 1,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("Sample count must be positive")

    def run(
        sampler: Callable[..., OnlineSample], rng: random.Random
    ) -> list[OnlineSample]:
        return [
            sampler(
                lm,
                grammar,
                rng=rng,
                pending_token_budget=pending_token_budget,
            )
            for _ in range(samples)
        ]

    action_outputs = run(sample_online_actions, random.Random(seed))
    token_outputs = run(sample_online_tokens, random.Random(seed))
    action = _summarize(
        lm,
        action_outputs,
        target_definition="locally normalized action-realization sequence mass",
    )
    token = _summarize(
        lm,
        token_outputs,
        target_definition="locally masked next-token probability",
    )
    return {
        "samples": samples,
        "seed": seed,
        "pending_token_budget": pending_token_budget,
        "action": action,
        "token": token,
        "comparison": {
            "target_call_ratio_token_over_action": (
                token["target_forward_passes"] / action["target_forward_passes"]
            ),
            "target_call_reduction": (
                1.0
                - action["target_forward_passes"] / token["target_forward_passes"]
            ),
            "accepted_tokens_per_call_ratio_action_over_token": (
                action["accepted_tokens_per_target_call"]
                / token["accepted_tokens_per_target_call"]
            ),
            "output_tokens_per_call_ratio_action_over_token": (
                action["output_tokens_per_target_call"]
                / token["output_tokens_per_target_call"]
            ),
            "latency_ratio_token_over_action": (
                token["latency_seconds"] / action["latency_seconds"]
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grammar", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=10)
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
    result = {
        "model": args.model,
        "revision": args.revision,
        "grammar": args.grammar.name,
        **run_online_comparison(
            lm,
            grammar,
            samples=args.samples,
            seed=args.seed,
            pending_token_budget=args.pending_token_budget,
        ),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())