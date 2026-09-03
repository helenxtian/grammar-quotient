# Grammar-Quotient Speculative Decoding

A speculative decoding framework that drafts and verifies **grammar-level macro
actions** — emitting a field, choosing an enum value, selecting a tool argument,
closing an object, expanding a grammar production — instead of individual tokens.

> **Core idea:** Rather than using the grammar only as a token-level filter, use
> the grammar to define the *units of speculation*.

---

## Problem Overview

LLMs increasingly produce structured outputs: JSON tool calls, SQL queries, API
arguments, workflow states, code fragments, agent actions. The output is not
merely free-form text — it must satisfy a formal constraint.

Let `p(w)` be the base LLM distribution over token strings `w ∈ V*`, and let
`L ⊆ V*` be the set of valid strings defined by a grammar, schema, or runtime
constraint system. The ideal structured-generation target is the
**grammar-conditioned distribution**:

```
p*(w) = p(w | w ∈ L) = p(w) · 1[w ∈ L] / p(L)
```

Most existing constrained decoders enforce validity by **masking invalid tokens**
at each step. This guarantees syntactic validity but does **not** generally
sample from `p(w | w ∈ L)`: each prefix-dependent renormalization changes the
relative probability of complete valid strings. The distortion is worst for
long-range constraints — nested JSON, balanced parentheses, quoted strings, SQL
scopes, function-call arguments, and dynamic constraints derived from long
context.

**Speculative decoding** adds a second layer. It accelerates inference with a
cheap draft model proposing tokens and a larger target model verifying them in
parallel. With the correct acceptance rule, samples match the target
distribution. But when output must satisfy a grammar, the intended target is no
longer the raw `p`, but the conditional `p*`. Locally masked speculative decoding
is faithful only to the *locally projected* distribution — not necessarily the
true grammar-conditioned one.

---

## The Gap

| Existing work | The question it asks |
|---|---|
| Grammar-constrained decoding (vLLM, SGLang, XGrammar, Outlines, Guidance, LLGuidance) | *Which next tokens are valid?* |
| Grammar-faithfulness theory (future validity / grammar-aligned decoding) | *How should token-level probabilities be corrected to match `p*`?* |
| **This project** | ***What if the grammar changes the unit of speculation itself?*** |

Instead of drafting and verifying token sequences, we draft and verify
**grammar-induced equivalence classes of token continuations**. This changes the
sample space of speculative decoding from raw tokens to structured grammar
actions.

**Central hypothesis:** structured outputs contain far fewer meaningful *grammar
choices* than *token-level choices*. Speculating over grammar actions lets the
decoder make multi-token progress per accepted step, reduce tokenizer-induced
artifacts, and gain a new theoretical route to grammar-conditioned correctness.

---

## Theory

### 1. Grammar-induced quotient space

Given the current token prefix `u` and grammar state `g`, the grammar permits a
set of next structured actions:

```
C(g) = { C₁, C₂, …, Cₘ }
```

Each `Cᵢ` is **not** a single token — it is a grammar-level macro action
corresponding to a set of token continuations `Cᵢ ⊆ V⁺`. Examples:

- `C₁`: emit JSON field `"name"`
- `C₂`: emit JSON field `"age"`
- `C₃`: choose enum value `"approved"`
- `C₄`: close current object
- `C₅`: choose SQL column `customer_id`

The grammar induces an equivalence relation over continuations: `s ∼_g s′` if
`s` and `s′` realize the same grammar action from `g`. The **quotient space** is:

```
Q_g = V⁺ / ∼_g
```

Standard speculative decoding proposes token sequences. Grammar-Quotient
Speculative Decoding proposes elements of `Q_g` — grammar actions.

### 2. Action-level probability mass

Base model mass on an action `C ∈ C(g)`:

```
P_p(C | u) = Σ_{s ∈ C} p(s | u)
```

For grammar-conditioned correctness, the mass must incorporate **future
validity**:

```
P_p*(C | u) ∝ Σ_{s ∈ C} p(s | u) · Φ(us)

   where  Φ(us) = Pr_{v ∼ p(·|us)} [ usv ∈ L ]
```

The quotient-level grammar-conditioned distribution over actions:

```
                Σ_{s ∈ C}  p(s | u) · Φ(us)
π_p*(C | u,g) = ──────────────────────────────────────
                Σ_{C′ ∈ C(g)} Σ_{s′ ∈ C′} p(s′ | u) · Φ(us′)
```

### 3. Action-level speculative verification

Let `q_Q(C | u,g)` be a cheap draft distribution over grammar actions. The draft
proposes a sequence `C₁, C₂, …, C_k`. The target verifier accepts each proposed
action with the **quotient-level acceptance rule**:

```
α(C, u, g) = min( 1,  π_p*(C | u,g) / π_q*(C | u,g) )
```

On acceptance, the decoder samples a concrete realization `s ∈ C`, advances the
prefix `u → us`, and updates grammar state `g → g′`.

### 4. Within-action realization

After accepting `C`, choose a concrete `s ∈ C`. Three cases:

- **Deterministic** — `C` has a canonical realization (fixed field name,
  delimiter): emit it directly.
- **Finite** — `C` is a finite set of aliases / enums / IDs / fields: sample
  ```
  p*(s | u, s ∈ C) = p(s|u)·Φ(us) / Σ_{s′ ∈ C} p(s′|u)·Φ(us′)
  ```
- **Open** — `C` is open-ended (free-form string value): fall back to token-level
  decoding until the action terminates.

### 5. Correctness claim

> If grammar actions are accepted according to the quotient-level target
> distribution `π_p*(C | u,g)`, and within-action realizations are sampled from
> the corresponding conditional, then the final output string is distributed
> according to `p(w | w ∈ L)`.

For a valid `w = s₁s₂⋯s_K` with `sᵢ ∈ Cᵢ`, the grammar-conditioned distribution
factorizes as:

```
p*(w) = Πᵢ  π_p*(Cᵢ | uᵢ, gᵢ) · p*(sᵢ | uᵢ, Cᵢ)
```

This separation of **structured choice** from **token realization** is the
theoretical basis for quotient-level speculative decoding.

---

## Why this may improve latency

Speedup arises when accepted grammar actions expand to multiple tokens. In
token-level speculation each accepted unit is one token; here each accepted unit
may be a field, enum value, SQL clause, JSON delimiter sequence, or tool
argument. The win holds when:

```
cost(verify action) < cost(verify all tokens in action)
```

**Expected to help most:** deterministic schema boilerplate, enum-heavy outputs,
tool calls, JSON records with fixed fields, SQL over known schemas, long-context
ID selection, agent state transitions.

**Expected to help less:** open-ended natural-language spans, where the grammar
offers little beyond "continue string."

---

## Research Questions

1. Can quotient-level speculative decoding be made **exactly correct** w.r.t.
   `p(w | w ∈ L)`?
2. For what grammar classes can action masses `Σ_{s ∈ C} p(s | u)` be computed
   **efficiently**?
3. When does action-level speculation **reduce target-model calls** vs.
   token-level?
4. Does quotient-level decoding **reduce tokenizer-induced bias** among enums,
   IDs, fields, and other structured candidates?
5. Can the method support **dynamic long-context constraints** (allowed IDs,
   available tools, DB columns, workflow states extracted from the prompt)?

---

## Expected Contribution

1. A **quotient-space formulation** of grammar-constrained generation.
2. A **speculative decoding rule** over grammar-induced macro actions.
3. A **correctness theorem** connecting quotient-level verification to the true
   grammar-conditioned distribution.

Unlike token-level grammar masking, this treats grammar as the space of
structured choices. Unlike standard speculative decoding, it verifies structured
actions rather than individual tokens. If successful, it provides a new path to
**faster and more faithful** structured LLM generation.

---

## Status

Early-stage research prototype. Implemented foundations include:

- the original exact synthetic-character correctness harness;
- finite phrase grammars with competing macro actions for reports, dialogue,
   Python code/docstrings, mathematical expressions, and tool-call JSON;
- an exact finite-language oracle and analytic quotient factorization; and
- finite-support speculative acceptance with positive-residual correction;
- a one-token pending frontier that repairs tokenizer-unstable action
   boundaries; and
- Qwen-backed exact-oracle quotient sampling over the 108-string dialogue
   benchmark.
- a reusable grammar-action generation engine with action traces, counters,
   unconstrained sampling, and finite-grammar token masking baselines.

Online local-target verification and an action-vs-token systems comparison are
implemented. A deterministic top-k/beam future-validity estimator is now
available for bounded phrase grammars, with reachable-state comparison against
the exact finite oracle. This estimator reports retained-path mass and is an
approximation whenever beam pruning removes valid paths; it does not by itself
establish exact online verification of the grammar-conditioned target.

### Tokenizer-boundary instrumentation

The phrase grammars can be scanned at every reachable action boundary with a
pinned tokenizer revision:

```bash
python -m gqsd.tokenizer_analysis \
   --model Qwen/Qwen2.5-0.5B \
   --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
   --pending-token-budget 1 \
   grammars/report_phrases.json \
   grammars/dialogue_phrases.json \
   grammars/code_docstring_phrases.json
```

For that Qwen revision, the original grammars contain 804 tokenizer-unstable
action boundaries: 6 in reports, 48 in dialogue, and 750 in code/docstrings.
Every observed crossing changes exactly one suffix token, so every case has a
canonical repair plan when the decoder keeps one token pending. Folding
deterministic literals into the following action removes all report crossings
and leaves 36 dialogue plus 720 code crossings between adjacent choices.

### Exact finite-oracle benchmark

The dialogue language can be scored exhaustively with pinned Qwen weights and
sampled using uniform action drafts, positive-residual correction, and the
pending-token frontier:

```bash
python -m gqsd.evaluate grammars/dialogue_phrases.json \
   --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
   --local-files-only \
   --batch-size 4 \
   --samples 100 \
   --seed 20260824
```

On the CPU run recorded for this revision, exact scoring of all 108 strings
took 9.997 seconds. The 100 sampled outputs had empirical TV 0.0317 from the
exact target; all 100 had canonical Qwen tokenizations and exercised boundary
repair, reclaiming 300 boundaries total. A baseline that commits every token
immediately rejected all 100 corresponding attempts at unstable boundaries.

This is an end-to-end finite-oracle reclamation result, not yet a production
speculative decoder: target probabilities are precomputed by exhaustive scoring,
so that experiment alone does not measure online target calls or latency. The
separate online benchmark below addresses the local-target systems comparison.

### Online action-vs-token verification

The online benchmark avoids exhaustive language scoring. At each grammar state,
action mode batches all competing realizations into one target forward pass;
token mode traverses the same candidate set through masked next-token decisions.
Both retain one pending token for boundary repair.

```bash
python -m gqsd.evaluate_online grammars/dialogue_phrases.json \
   --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
   --local-files-only \
   --samples 10 \
   --seed 20260824
```

Pinned Qwen CPU results:

| Metric | Action | Token |
|---|---:|---:|
| Valid canonical outputs | 10/10 | 10/10 |
| Target forward passes | 50 | 227 |
| Calls per sample | 5.0 | 22.7 |
| Accepted tokens per call | 2.18 | 0.934 |
| Output tokens per call | 3.38 | 0.780 |
| Total latency | 13.94 s | 36.50 s |
| Reclaimed boundaries | 30 | 30 |

Action verification used 4.54x fewer target calls and measured 2.62x lower
latency in this run. These are small CPU benchmark results, not production
throughput claims; larger repeated runs and accelerator measurements remain.

The targets also differ intentionally. Action mode normalizes base-model
sequence mass over the current grammar realizations, while token mode performs
standard local next-token masking. Without an online future-validity term
`Phi`, neither comparison establishes exact preservation of `p(w | w in L)`.

### Approximate online future validity

`gqsd.phi.BeamPhiEstimator` estimates `Phi(u)` by expanding grammar action
realizations with model sequence probabilities and retaining a fixed beam at
each depth. The retained terminal mass is a lower bound on the mass found by
the search. It can be supplied to `sample_online_actions` with the
`phi_estimator` argument; action weights then use

```
Σ_s p(s | u) · Phi_hat(us)
```

The `compare_phi` helper evaluates absolute log-ratio error and support loss
over every reachable state of a finite grammar against `FiniteGrammarOracle`.
Boundary-unstable intermediate actions use the existing pending-frontier
reclamation path; exact boundary accounting still requires passing that
frontier context into a future rollout.

For local error `epsilon` over a grammar horizon `H`, the evaluator uses the
finite-horizon bound `tanh(H * epsilon)` for terminal TV. On the pinned Qwen
dialogue grammar, the beam sweep produced terminal TV of 0.00741, 0.0000024,
0.0000018, and 0.0000018 for beam sizes 1, 2, 4, and 8 respectively. Beam 4
had mean absolute log-ratio error 0.000034 and maximum error 0.00230, giving a
conservative finite-horizon TV bound of 0.0115. These are finite-model
measurements, not generalization claims; larger sweeps need KV-cache or prefix
reuse because CPU long-context forward passes remain expensive.

The model wrapper now includes shared-prefix cache scoring. It is opt-in in
`generate_actions` and the benchmark CLI because the first pinned CPU dialogue
run reduced context work but increased wall-clock time: 10 short cache forwards
versus 5 padded forwards, and 2.07s versus 1.08s for one action sample. Models
that cannot clone or advance their cache fall back to the regular padded batch
scorer.

To compare the three decoder families on an enumerable grammar, run:

```bash
.venv/bin/python -m gqsd.evaluate_decoders grammars/dialogue_phrases.json \
   --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
   --local-files-only --samples 100
```

Add `--prefix-cache` to measure the experimental cache path explicitly.

The JSON report includes validity rate, conditional TV/KL to exact `p*`,
target forward passes, and latency for unconstrained, token-masked, padded
action, cache action, and adaptive action decoding. TV/KL are calculated over
valid outputs; invalid outputs are retained in the separately reported
validity rate. Add `--output results/dialogue-100.json` to save the report.

Finite phrase grammars can also include bounded open spans. The action engine
then samples tokens until the span's explicit stop marker or token budget is
reached. These hybrid outputs are not finite-oracle samples, so exact `p*`
comparison remains limited to fully enumerable grammars.
