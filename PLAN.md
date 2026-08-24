# Project Plan

Two semi-independent tracks: **theory** (prove the correctness theorem,
characterize `Φ`) and **systems** (prototype, measure speedup). The dominant
risk is `Φ(us)` — future validity — which is `#P`-hard in general. The first real
result to chase is: *on the toy grammar, can we compute exact `Φ` and does the
quotient sampler provably hit `p*`?*

## Phase 0 — Scaffolding ✅
- [x] `git init`, repo on GitHub (`helenxtian/grammar-quotient`, private)
- [x] Python package layout (`pyproject.toml`, `src/gqsd/`, `tests/`, `grammars/`)
- [x] Base model wrapper with next-token logits + sequence scoring (`model.py`,
      default `Qwen/Qwen2.5-0.5B`)
- [x] Minimal grammar substrate decided: **fixed-schema JSON object**, finite and
      fully enumerable (`grammars/toy_record.json`, `grammar.py` interfaces)
- [x] Smoke tests

## Phase 1 — Baselines + exact oracle
- [ ] Unconstrained sampling from `p` (`baselines.sample_unconstrained`)
- [ ] Token-level masked decoding — the biased baseline
      (`baselines.sample_token_masked`)
- [x] Brute-force exact `p*` oracle over finite grammars (`oracle.py`)
- [ ] First experiment: show masked decoding ≠ `p*` (TV/KL gap) on the toy schema

## Phase 2 — Quotient mechanics (no speculation)
- [x] Finite phrase `GrammarState`: `actions()`, `advance()`, `is_accepting()`
- [x] Exact action and realization mass via finite-language lookahead
- [x] Exact `Φ` through exhaustive finite-language oracle lookahead
- [x] **Synthetic validation gate:** analytic quotient factorization matches the
      exact finite-language `p*` oracle.
- [x] **Qwen validation gate:** 100 speculative dialogue samples use canonical
      tokenization and have empirical TV 0.0317 from the exact 108-string target.

## Phase 3 — Speculation
- [x] Uniform cheap draft `q_Q` over actions and within-action realizations
- [x] Finite-support acceptance rule plus exact positive-residual correction
- [x] Re-run sampled TV-to-oracle gate with speculation ON
- [x] Integrate one-token pending frontier; reclaim 300 boundaries across 100
      Qwen-backed dialogue samples that strict immediate commitment rejects
- [ ] Measure tokens-accepted-per-target-call vs. token-level speculative

## Phase 4 — Scale + evaluation
- [ ] Nested JSON, tool-call schemas, SQL-over-fixed-schema, enum-heavy outputs
- [ ] Metrics: faithfulness vs. masking, target-calls/latency, validity rate,
      tokenizer-bias among enums/IDs (RQ4)
- [ ] Dynamic long-context constraints (RQ5)

## Theory track (parallel)
- [ ] Formal proof of the Phase-2 factorization; should agree with the KL test
- [ ] Characterize the target under **approximate** `Φ` — decide the framing:
      "exactly correct given a `Φ` oracle" vs. "correct + tractable `Φ` approx".

## Current implementation note

The repository now includes exactly enumerable phrase grammars for controlled
reports, dialogue, and Python code/docstrings. These establish non-vacuous,
competing grammar actions while keeping exact terminal normalization possible.
Real Qwen scoring, tokenizer-mismatch instrumentation, a uniform action draft
loop, and pending-token reclamation are now integrated for the finite dialogue
benchmark. Approximate `Φ`, online target verification, and performance
evaluation remain required before speed claims are supported.

Tokenizer-boundary instrumentation is now implemented and validated against
Qwen/Qwen2.5-0.5B revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.
All 804 observed crossings require a one-token pending suffix, and the scanner
constructs canonical boundary-repair plans. The exact finite-oracle dialogue
benchmark exercises 300 such repairs over 100 samples while preserving canonical
Qwen tokenization; strict immediate token commitment rejects all 100 attempts.
