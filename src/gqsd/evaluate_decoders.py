"""Compare unconstrained, token-masked, and grammar-action decoding."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .baselines import sample_token_masked, sample_unconstrained
from .generate import generate_actions
from .model import DEFAULT_MODEL, LM
from .oracle import FiniteGrammarOracle
from .phrase_grammar import PhraseGrammar


def _tv(empirical: dict[str, float], target: dict[str, float]) -> float:
    return 0.5 * sum(abs(empirical.get(key, 0.0) - value) for key, value in target.items())


def _kl(empirical: dict[str, float], target: dict[str, float]) -> float:
    return sum(
        probability * math.log(probability / target[key])
        for key, probability in empirical.items()
        if probability > 0.0 and key in target
    )


def _summarize(
    name: str,
    outputs: list[str | None],
    oracle: FiniteGrammarOracle,
    forwards: list[int],
    latencies: list[float],
) -> dict[str, Any]:
    valid = [output for output in outputs if output in oracle.p_star]
    counts = Counter(valid)
    empirical = {text: counts[text] / len(valid) for text in oracle.p_star} if valid else {}
    return {
        "decoder": name,
        "samples": len(outputs),
        "valid": len(valid),
        "validity_rate": len(valid) / len(outputs),
        "total_variation_to_p_star": _tv(empirical, oracle.p_star) if valid else None,
        "kl_to_p_star": _kl(empirical, oracle.p_star) if valid else None,
        "target_forward_passes": sum(forwards),
        "target_forward_passes_per_sample": sum(forwards) / len(outputs),
        "latency_seconds": sum(latencies),
        "latency_seconds_per_sample": sum(latencies) / len(outputs),
        "example": outputs[0],
    }


def run_decoder_benchmark(
    lm: LM,
    grammar: PhraseGrammar,
    oracle: FiniteGrammarOracle,
    *,
    samples: int,
    seed: int,
    max_tokens: int,
    use_prefix_cache: bool = True,
) -> list[dict[str, Any]]:
    if samples <= 0:
        raise ValueError("Sample count must be positive")
    if max_tokens < 0:
        raise ValueError("Maximum token count must be nonnegative")

    methods: list[tuple[str, Callable[[random.Random], str | None]]] = [
        (
            "unconstrained",
            lambda rng: sample_unconstrained(lm, "", max_tokens, rng=rng),
        ),
        (
            "token_masked",
            lambda rng: sample_token_masked(
                lm, "", grammar.start(), max_tokens, rng=rng
            ),
        ),
        (
            "grammar_actions",
            lambda rng: generate_actions(
                lm, grammar, rng=rng, use_prefix_cache=use_prefix_cache
            ).text,
        ),
    ]
    results = []
    for name, decode in methods:
        outputs: list[str | None] = []
        forwards: list[int] = []
        latencies: list[float] = []
        for offset in range(samples):
            started = time.perf_counter()
            lm.reset_counters()
            try:
                output = decode(random.Random(seed + offset))
            except ValueError:
                output = None
            outputs.append(output)
            forwards.append(lm.target_forward_passes)
            latencies.append(time.perf_counter() - started)
        results.append(_summarize(name, outputs, oracle, forwards, latencies))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grammar", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--prefix-cache", action="store_true")
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
        "grammar": grammar.name,
        "support_size": len(oracle.p_star),
        "score_seconds": score_seconds,
        "p_star_sum": math.fsum(oracle.p_star.values()),
        "decoders": run_decoder_benchmark(
            lm,
            grammar,
            oracle,
            samples=args.samples,
            seed=args.seed,
            max_tokens=args.max_tokens,
            use_prefix_cache=args.prefix_cache,
        ),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())