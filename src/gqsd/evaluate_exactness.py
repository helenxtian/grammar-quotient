"""Compare grammar-action sampling with and without exact future validity."""

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

from .baselines import sample_token_masked
from .generate import generate_actions
from .model import DEFAULT_MODEL, LM
from .oracle import FiniteGrammarOracle
from .phi import BeamPhiEstimator, ExactPhiEstimator
from .phrase_grammar import PhraseGrammar
from .quotient import sample_speculative_quotient


def _summary(
    name: str,
    outputs: list[str | None],
    oracle: FiniteGrammarOracle,
    forwards: list[int],
    latencies: list[float],
) -> dict[str, Any]:
    valid = [output for output in outputs if output in oracle.p_star]
    counts = Counter(valid)
    empirical = {
        text: counts[text] / len(valid) for text in oracle.p_star
    } if valid else {}
    tv = (
        0.5 * sum(abs(empirical.get(text, 0.0) - probability)
                   for text, probability in oracle.p_star.items())
        if valid else None
    )
    kl = (
        sum(
            probability * math.log(probability / oracle.p_star[text])
            for text, probability in empirical.items()
            if probability > 0.0
        )
        if valid else None
    )
    return {
        "decoder": name,
        "samples": len(outputs),
        "valid": len(valid),
        "validity_rate": len(valid) / len(outputs),
        "total_variation_to_p_star": tv,
        "kl_to_p_star": kl,
        "target_forward_passes": sum(forwards),
        "target_forward_passes_per_sample": sum(forwards) / len(outputs),
        "latency_seconds": sum(latencies),
        "latency_seconds_per_sample": sum(latencies) / len(outputs),
        "example": outputs[0],
    }


def run_exactness_benchmark(
    lm: LM,
    grammar: PhraseGrammar,
    oracle: FiniteGrammarOracle,
    *,
    samples: int,
    seed: int,
    max_tokens: int,
    batch_size: int = 32,
    beam_size: int = 8,
) -> list[dict[str, Any]]:
    """Compare exact, local, exact-Phi, beam-Phi, and masked decoding."""
    if samples <= 0:
        raise ValueError("Sample count must be positive")
    if max_tokens < 0:
        raise ValueError("Maximum token count must be nonnegative")
    if beam_size <= 0:
        raise ValueError("Beam size must be positive")
    exact_phi = ExactPhiEstimator(oracle, lm)
    beam_phi = BeamPhiEstimator(
        lm, grammar, beam_size=beam_size, score_batch_size=batch_size
    )
    methods: list[tuple[str, Callable[[random.Random], str | None]]] = [
        (
            "exact_quotient",
            lambda rng: sample_speculative_quotient(
                oracle, lm.tokenizer, rng=rng
            ).text,
        ),
        (
            "grammar_actions_no_phi",
            lambda rng: generate_actions(lm, grammar, rng=rng).text,
        ),
        (
            "grammar_actions_exact_phi",
            lambda rng: generate_actions(
                lm, grammar, rng=rng, phi_estimator=exact_phi
            ).text,
        ),
        (
            "grammar_actions_beam_phi",
            lambda rng: generate_actions(
                lm, grammar, rng=rng, phi_estimator=beam_phi
            ).text,
        ),
        (
            "token_masked",
            lambda rng: sample_token_masked(
                lm, "", grammar.start(), max_tokens, rng=rng
            ),
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
        summary = _summary(name, outputs, oracle, forwards, latencies)
        if name == "grammar_actions_beam_phi":
            summary["phi_score_batches"] = beam_phi.score_batches
            summary["phi_scored_texts"] = beam_phi.scored_texts
        results.append(summary)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grammar", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output", type=Path)
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
    result = {
        "model": args.model,
        "revision": args.revision,
        "grammar": grammar.name,
        "support_size": len(oracle.p_star),
        "score_seconds": time.perf_counter() - started,
        "decoders": run_exactness_benchmark(
            lm,
            grammar,
            oracle,
            samples=args.samples,
            seed=args.seed,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            beam_size=args.beam_size,
        ),
    }
    report = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())