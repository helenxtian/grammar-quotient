"""Measure beam-based future-validity error on an enumerable phrase grammar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .model import DEFAULT_MODEL, LM
from .oracle import FiniteGrammarOracle
from .phi import (
    BeamPhiEstimator,
    approximate_quotient_distribution,
    compare_phi,
    distribution_kl,
    distribution_tv,
    finite_horizon_tv_bound,
    reachable_states,
)
from .phrase_grammar import PhraseGrammar


def run_phi_benchmark(
    grammar: PhraseGrammar,
    lm: LM,
    *,
    beam_sizes: list[int],
    batch_size: int = 32,
) -> dict[str, Any]:
    """Return reproducible state and terminal metrics for each beam size."""
    oracle = FiniteGrammarOracle.from_lm(grammar, lm, batch_size=batch_size)
    horizon = max(state.segment_index for state in reachable_states(grammar))
    results: list[dict[str, Any]] = []
    for beam_size in beam_sizes:
        estimator = BeamPhiEstimator(
            lm, grammar, beam_size=beam_size, score_batch_size=batch_size
        )
        lm.reset_counters()
        comparison = compare_phi(estimator, oracle, lm)
        state_forward_passes = lm.target_forward_passes
        lm.reset_counters()
        distribution = approximate_quotient_distribution(estimator, grammar)
        results.append(
            {
                "beam_size": beam_size,
                "state_count": comparison.states,
                "mean_absolute_log_ratio": comparison.mean_absolute_log_ratio,
                "maximum_absolute_log_ratio": comparison.maximum_absolute_log_ratio,
                "support_loss": comparison.support_loss,
                "exact_zero_mass": comparison.exact_zero_mass,
                "terminal_kl": distribution_kl(distribution, oracle.p_star),
                "terminal_tv": distribution_tv(distribution, oracle.p_star),
                "finite_horizon_tv_bound": finite_horizon_tv_bound(
                    horizon, comparison.maximum_absolute_log_ratio
                ),
                "state_forward_passes": state_forward_passes,
                "distribution_forward_passes": lm.target_forward_passes,
            }
        )
    return {
        "grammar": grammar.name,
        "support_size": len(oracle.p_star),
        "horizon": horizon,
        "beam_sizes": beam_sizes,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grammar", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--beam-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if any(beam_size <= 0 for beam_size in args.beam_sizes):
        parser.error("beam sizes must be positive")

    grammar = PhraseGrammar.load(args.grammar)
    lm = LM.load(
        args.model,
        device=args.device,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    result = run_phi_benchmark(
        grammar,
        lm,
        beam_sizes=args.beam_sizes,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())