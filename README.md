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

Early-stage research proposal. See the project plan for the roadmap from theory
to prototype to evaluation.
