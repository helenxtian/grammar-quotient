"""Generate one sample from a finite phrase grammar."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .baselines import sample_token_masked, sample_unconstrained
from .generate import generate_actions
from .model import DEFAULT_MODEL, LM
from .oracle import FiniteGrammarOracle
from .phi import BeamPhiEstimator, ExactPhiEstimator
from .phrase_grammar import PhraseGrammar


def generate_one(
    lm: LM,
    grammar: PhraseGrammar,
    *,
    mode: str,
    prompt: str,
    seed: int,
    max_tokens: int,
    cache_policy: str,
    beam_size: int,
    oracle: FiniteGrammarOracle | None = None,
) -> dict[str, Any]:
    """Generate one sample and return JSON-compatible result data."""
    if mode not in {"unconstrained", "token_masked", "grammar_actions", "exact_phi", "beam_phi"}:
        raise ValueError("Unknown generation mode")
    if cache_policy not in {"padded", "cache", "adaptive"}:
        raise ValueError("Cache policy must be padded, cache, or adaptive")
    if mode in {"exact_phi", "beam_phi"} and oracle is None:
        raise ValueError("This mode requires a finite grammar oracle")

    rng = random.Random(seed)
    lm.reset_counters()
    started = time.perf_counter()
    if mode == "unconstrained":
        text = sample_unconstrained(lm, prompt, max_tokens, rng=rng)
        result: dict[str, Any] = {"text": text}
    elif mode == "token_masked":
        text = sample_token_masked(
            lm, prompt, grammar.start(), max_tokens, rng=rng
        )
        result = {"text": text}
    else:
        phi_estimator = None
        if mode == "exact_phi":
            phi_estimator = ExactPhiEstimator(oracle, lm)
        elif mode == "beam_phi":
            phi_estimator = BeamPhiEstimator(
                lm, grammar, beam_size=beam_size
            )
        generation = generate_actions(
            lm,
            grammar,
            prompt=prompt,
            rng=rng,
            phi_estimator=phi_estimator,
            cache_policy=cache_policy,
        )
        result = asdict(generation)
        if isinstance(phi_estimator, BeamPhiEstimator):
            result["phi_score_batches"] = phi_estimator.score_batches
            result["phi_scored_texts"] = phi_estimator.scored_texts

    result.update(
        {
            "mode": mode,
            "grammar": grammar.name,
            "seed": seed,
            "target_forward_passes": lm.target_forward_passes,
            "latency_seconds": time.perf_counter() - started,
        }
    )
    if oracle is not None:
        output = result["text"]
        result["oracle_valid"] = output in oracle.p_star
        result["support_size"] = len(oracle.p_star)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grammar", type=Path)
    parser.add_argument(
        "--mode",
        choices=("unconstrained", "token_masked", "grammar_actions", "exact_phi", "beam_phi"),
        default="grammar_actions",
    )
    parser.add_argument("--prompt", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--cache-policy", choices=("padded", "cache", "adaptive"), default="padded")
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    grammar = PhraseGrammar.load(args.grammar)
    lm = LM.load(
        args.model,
        device=args.device,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    oracle = (
        None
        if args.mode == "unconstrained"
        else FiniteGrammarOracle.from_lm(grammar, lm, batch_size=args.batch_size)
    )
    report = generate_one(
        lm,
        grammar,
        mode=args.mode,
        prompt=args.prompt,
        seed=args.seed,
        max_tokens=args.max_tokens,
        cache_policy=args.cache_policy,
        beam_size=args.beam_size,
        oracle=oracle,
    )
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
