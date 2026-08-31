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

- [x] Unconstrained sampling from `p` (`baselines.sample_unconstrained`)
- [x] Token-level masked decoding — the biased baseline
      (`baselines.sample_token_masked`)
- [x] Brute-force exact `p*` oracle over finite grammars (`oracle.py`)
- [ ] First experiment: show masked decoding != `p*` (TV/KL gap) on the toy schema

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
- [x] Online local-target benchmark: batch competing action realizations in one
      target call and compare against token-trie speculative verification
- [x] Measure target calls, accepted/output tokens per call, and CPU latency:
      50 vs. 227 calls and 13.94s vs. 36.50s over 10 dialogue samples
- [x] Add deterministic beam-based online `Phi` estimation/correction; expose
      state-level log-ratio and support-loss comparison against the exact oracle

## Phase 4 — Scale + evaluation
- [ ] Nested JSON, tool-call schemas, SQL-over-fixed-schema, enum-heavy outputs
- [x] Metrics: faithfulness vs. masking, target-calls/latency, validity rate,
      tokenizer-bias among enums/IDs (RQ4)
- [ ] Dynamic long-context constraints (RQ5)

## Theory track (parallel)
- [ ] Formal proof of the Phase-2 factorization; should agree with the KL test
- [~] Characterize the target under **approximate** `Φ` — the implementation
      now frames beam mass as an explicit approximation, but terminal KL/TV
      bounds and broader estimators remain:
      "exactly correct given a `Φ` oracle" vs. "correct + tractable `Φ` approx".
- [x] Add finite-horizon TV bound and finite-grammar terminal KL/TV evaluator
- [x] Add KV-cache/prefix reuse before running larger real-Qwen beam sweeps;
- [x] Add opt-in model prefix-cache scoring primitive with regression coverage
- [x] Integrate cache reuse into batched action scoring for branchable models;
      cache cloning falls back to the regular padded batch scorer

## Phase 5 - Usable generation surface
- [x] Reusable `generate_actions` engine with action trace and counters
- [x] Initial unconstrained and finite-grammar token-masked baselines
- [x] Integrate prefix cache reuse into batched action scoring
- [x] Common decoder benchmark with validity, TV/KL-to-oracle, target work,
      and latency metrics
- [x] Bounded open-span fallback with token-level stop-marker decoding
- [ ] Add CLI generation command and common benchmark report

## Current implementation note

The repository now includes exactly enumerable phrase grammars for controlled
reports, dialogue, and Python code/docstrings. These establish non-vacuous,
competing grammar actions while keeping exact terminal normalization possible.
Real Qwen scoring, tokenizer-mismatch instrumentation, a uniform action draft
loop, pending-token reclamation, and deterministic approximate `Φ` estimation
are now integrated for the finite dialogue benchmark. Online verification of
`p*`, accelerator benchmarks, and broader performance evaluation remain
required before general speed or faithfulness claims are supported.

Online local-target verification is now also implemented. On the pinned
10-sample CPU run, action batching used 4.54x fewer target forwards and 2.62x
lower latency than token-trie verification. This comparison does not use exact
future validity: action and token modes target their respective locally
normalized distributions, so exact online faithfulness to `p*` remains open.

Tokenizer-boundary instrumentation is now implemented and validated against
Qwen/Qwen2.5-0.5B revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.
All 804 observed crossings require a one-token pending suffix, and the scanner
constructs canonical boundary-repair plans. The exact finite-oracle dialogue
benchmark exercises 300 such repairs over 100 samples while preserving canonical
Qwen tokenization; strict immediate token commitment rejects all 100 attempts.
