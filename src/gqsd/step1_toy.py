"""Step 1 exact-correctness harness on a fully enumerable toy grammar.

This module intentionally avoids a real LLM. It defines a synthetic
autoregressive next-token model with deterministic pseudo-random conditional
probabilities, so we can:

1. Enumerate the full language L exactly (2000 strings).
2. Compute p*(w) = p(w) / sum_{w' in L} p(w') exactly.
3. Compute Phi(u) exactly by summing over valid completions.
4. Compare empirical samplers against exact ground truth via KL.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

STATUS_VALUES: tuple[str, ...] = ('"approved"', '"denied"')
DIGITS: tuple[str, ...] = tuple(str(i) for i in range(10))
PREFIX_STATUS = '{"status": '
MIDDLE_ID = ', "id": '
SUFFIX_OBJ = '}'


def enumerate_all_valid_strings() -> list[str]:
    """Return all valid strings in the toy grammar language L (size 2000)."""
    out: list[str] = []
    for status in STATUS_VALUES:
        for a in DIGITS:
            for b in DIGITS:
                for c in DIGITS:
                    out.append(f"{PREFIX_STATUS}{status}{MIDDLE_ID}{a}{b}{c}{SUFFIX_OBJ}")
    return out


@dataclass(frozen=True)
class CompiledAction:
    """A grammar action C(g) with a finite set of concrete realizations."""

    label: str
    kind: str  # deterministic | finite
    realizations: tuple[str, ...]


def compile_actions(id_granularity: str = "three_digits") -> list[CompiledAction]:
    """Compile the toy grammar into tagged grammar actions.

    Args:
        id_granularity: "three_digits" for 3x10 finite decisions, or "id_1000"
            for one 1000-way finite decision.
    """
    if id_granularity not in {"three_digits", "id_1000"}:
        raise ValueError("id_granularity must be 'three_digits' or 'id_1000'")

    actions: list[CompiledAction] = [
        CompiledAction("emit_prefix", "deterministic", (PREFIX_STATUS,)),
        CompiledAction("choose_status", "finite", STATUS_VALUES),
        CompiledAction("emit_id_prefix", "deterministic", (MIDDLE_ID,)),
    ]

    if id_granularity == "three_digits":
        digit_choices = tuple(DIGITS)
        actions.extend(
            [
                CompiledAction("choose_id_d1", "finite", digit_choices),
                CompiledAction("choose_id_d2", "finite", digit_choices),
                CompiledAction("choose_id_d3", "finite", digit_choices),
            ]
        )
    else:
        ids = tuple(f"{a}{b}{c}" for a in DIGITS for b in DIGITS for c in DIGITS)
        actions.append(CompiledAction("choose_id", "finite", ids))

    actions.append(CompiledAction("emit_suffix", "deterministic", (SUFFIX_OBJ,)))
    return actions


class SyntheticToyModel:
    """Deterministic synthetic autoregressive model over characters.

    The next-token probabilities are generated from a stable hash of
    (seed, prefix, token), so p(token | prefix) is cheap and reproducible.
    """

    def __init__(self, vocabulary: Iterable[str], seed: int = 0) -> None:
        self.vocabulary = tuple(sorted(set(vocabulary)))
        self.seed = seed
        self._probs_cache: dict[str, dict[str, float]] = {}

    def _score(self, prefix: str, token: str) -> float:
        payload = f"{self.seed}|{prefix}|{token}".encode()
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        # Strictly positive score in (0, 1], then shifted for better spread.
        val = int.from_bytes(digest, byteorder="big")
        return (val + 1) / float(2**64)

    def next_token_probs(self, prefix: str) -> dict[str, float]:
        cached = self._probs_cache.get(prefix)
        if cached is not None:
            return cached
        raw = {tok: self._score(prefix, tok) for tok in self.vocabulary}
        z = sum(raw.values())
        probs = {tok: s / z for tok, s in raw.items()}
        self._probs_cache[prefix] = probs
        return probs

    def next_token_prob(self, prefix: str, token: str) -> float:
        probs = self.next_token_probs(prefix)
        return probs[token]

    def sequence_prob(self, text: str) -> float:
        p = 1.0
        prefix = ""
        for ch in text:
            p *= self.next_token_prob(prefix, ch)
            prefix += ch
        return p

    def continuation_prob(self, prefix: str, continuation: str) -> float:
        p = 1.0
        cur = prefix
        for ch in continuation:
            p *= self.next_token_prob(cur, ch)
            cur += ch
        return p


class _TrieNode:
    """Node in the prefix trie compiled from the grammar's language L."""

    __slots__ = ("children", "terminal")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.terminal: bool = False


def build_trie(strings: Iterable[str]) -> _TrieNode:
    """Compile a set of strings into a prefix trie (the PDA-over-trie for L).

    Each root-to-node path is a valid prefix; terminal nodes mark complete
    strings. This is the acceptance structure the token/quotient samplers walk.
    """
    root = _TrieNode()
    for s in strings:
        node = root
        for ch in s:
            node = node.children.setdefault(ch, _TrieNode())
        node.terminal = True
    return root


@dataclass
class ToyOracle:
    """Exact oracles for p*(w) and Phi(u), backed by a compiled prefix trie."""

    model: SyntheticToyModel
    valid_strings: list[str]

    def __post_init__(self) -> None:
        self._valid_set = set(self.valid_strings)
        self._root = build_trie(self.valid_strings)
        self._raw_probs = {w: self.model.sequence_prob(w) for w in self.valid_strings}
        z = sum(self._raw_probs.values())
        self.p_star = {w: p / z for w, p in self._raw_probs.items()}
        self._phi_cache: dict[str, float] = {}

    def _find_node(self, prefix: str) -> _TrieNode | None:
        node = self._root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def valid_next_tokens(self, prefix: str) -> tuple[str, ...]:
        node = self._find_node(prefix)
        if node is None:
            return ()
        return tuple(sorted(node.children))

    def is_complete(self, prefix: str) -> bool:
        return prefix in self._valid_set

    def phi(self, prefix: str) -> float:
        """Exact Phi(u): total base-model mass of valid completions of ``prefix``.

        Computed recursively over the trie: a terminal node contributes the
        empty completion (mass 1), and each child contributes
        p(tok | prefix) * Phi(prefix + tok). This telescopes so that Phi("")
        equals p(L) exactly.
        """
        node = self._find_node(prefix)
        if node is None:
            return 0.0
        return self._phi_node(node, prefix)

    def _phi_node(self, node: _TrieNode, prefix: str) -> float:
        cached = self._phi_cache.get(prefix)
        if cached is not None:
            return cached
        total = 1.0 if node.terminal else 0.0
        for tok, child in node.children.items():
            total += self.model.next_token_prob(prefix, tok) * self._phi_node(child, prefix + tok)
        self._phi_cache[prefix] = total
        return total


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    z = sum(weights.values())
    if z <= 0.0:
        raise ValueError("Cannot normalize non-positive total mass")
    return {k: v / z for k, v in weights.items()}


def _sample_from_dist(rng: random.Random, dist: dict[str, float]) -> str:
    r = rng.random()
    acc = 0.0
    last_key = ""
    for key, prob in dist.items():
        acc += prob
        last_key = key
        if r <= acc:
            return key
    return last_key


def _speculative_pick(
    rng: random.Random,
    target: dict[str, float],
    draft: dict[str, float],
    stats: dict[str, int] | None = None,
) -> str:
    """One-step speculative pick with exact residual fallback.

    Draft y ~ q, accept with prob min(1, p(y)/q(y)); on rejection resample from
    the normalized positive residual (p - q)+. This preserves the exact target
    distribution over a finite support for any draft q (the §3.3 invariance).
    ``stats`` optionally accumulates {"trials", "accepts"} to expose the
    acceptance rate.
    """
    if stats is not None:
        stats["trials"] += 1
    y = _sample_from_dist(rng, draft)
    qy = draft.get(y, 0.0)
    py = target.get(y, 0.0)
    if qy <= 0.0:
        accept_prob = 1.0 if py > 0.0 else 0.0
    else:
        accept_prob = min(1.0, py / qy)
    if rng.random() <= accept_prob:
        if stats is not None:
            stats["accepts"] += 1
        return y

    residual = {k: max(0.0, target.get(k, 0.0) - draft.get(k, 0.0)) for k in target}
    z = sum(residual.values())
    if z <= 1e-15:
        return _sample_from_dist(rng, target)
    residual = {k: v / z for k, v in residual.items() if v > 0.0}
    return _sample_from_dist(rng, residual)


def _draft_distribution(keys: Iterable[str], mode: str, target: dict[str, float]) -> dict[str, float]:
    keys_tuple = tuple(keys)
    if mode == "target":
        return dict(target)
    if mode == "uniform":
        p = 1.0 / len(keys_tuple)
        return {k: p for k in keys_tuple}
    raise ValueError("draft mode must be 'target' or 'uniform'")


def sample_token_level(
    oracle: ToyOracle,
    *,
    use_phi: bool,
    draft_mode: str,
    rng: random.Random,
    stats: dict[str, int] | None = None,
) -> str:
    """Sample one valid string with token-level constrained decoding.

    - use_phi=False: plain token masking baseline.
    - use_phi=True: token-level Phi correction.
    """
    prefix = ""
    while True:
        if prefix in oracle._valid_set:
            return prefix

        valid_tokens = oracle.valid_next_tokens(prefix)
        if not valid_tokens:
            raise RuntimeError("Reached dead prefix while token-level sampling")

        if use_phi:
            weights = {
                tok: oracle.model.next_token_prob(prefix, tok) * oracle.phi(prefix + tok)
                for tok in valid_tokens
            }
        else:
            weights = {tok: oracle.model.next_token_prob(prefix, tok) for tok in valid_tokens}

        target = _normalize(weights)
        draft = _draft_distribution(valid_tokens, draft_mode, target)
        tok = _speculative_pick(rng, target, draft, stats)
        prefix += tok


def _action_mass(
    oracle: ToyOracle,
    prefix: str,
    action: CompiledAction,
    use_phi: bool,
) -> float:
    total = 0.0
    for s in action.realizations:
        p = oracle.model.continuation_prob(prefix, s)
        if use_phi:
            p *= oracle.phi(prefix + s)
        total += p
    return total


def sample_quotient_level(
    oracle: ToyOracle,
    actions: list[CompiledAction],
    *,
    use_phi: bool,
    draft_mode: str,
    rng: random.Random,
    stats: dict[str, int] | None = None,
) -> str:
    """Sample one valid string with quotient-level action decoding.

    Uses finite-action speculative verification plus within-action realization.
    """
    prefix = ""
    for action in actions:
        action_weights: dict[str, float] = {action.label: _action_mass(oracle, prefix, action, use_phi)}

        # In this toy compiler each step has one action label but potentially many
        # realizations; keep API aligned with general action-level distributions.
        target_action = _normalize(action_weights)
        draft_action = _draft_distribution(target_action.keys(), draft_mode, target_action)
        chosen_label = _speculative_pick(rng, target_action, draft_action, stats)
        if chosen_label != action.label:
            raise RuntimeError("Unexpected action label mismatch")

        if len(action.realizations) == 1:
            s = action.realizations[0]
        else:
            if use_phi:
                within = {
                    s: oracle.model.continuation_prob(prefix, s) * oracle.phi(prefix + s)
                    for s in action.realizations
                }
            else:
                within = {s: oracle.model.continuation_prob(prefix, s) for s in action.realizations}
            target_within = _normalize(within)
            draft_within = _draft_distribution(action.realizations, draft_mode, target_within)
            s = _speculative_pick(rng, target_within, draft_within, stats)
        prefix += s

    if prefix not in oracle._valid_set:
        raise RuntimeError("Quotient sampler produced invalid string")
    return prefix


def empirical_distribution(samples: list[str], support: list[str]) -> dict[str, float]:
    counts = Counter(samples)
    n = float(len(samples))
    return {w: counts.get(w, 0) / n for w in support}


def analytic_token_level_distribution(
    oracle: ToyOracle, *, use_phi: bool
) -> dict[str, float]:
    """Exact (noise-free) induced distribution of the token-level sampler over L.

    Walks the trie once per string, multiplying the locally-normalized per-token
    probabilities. With ``use_phi=True`` this equals p* exactly; with
    ``use_phi=False`` it is the plain masking distribution (generally != p*).
    """
    model = oracle.model
    dist: dict[str, float] = {}
    for w in oracle.valid_strings:
        prob = 1.0
        prefix = ""
        node = oracle._root
        for ch in w:
            children = node.children
            if use_phi:
                weights = {
                    tok: model.next_token_prob(prefix, tok) * oracle.phi(prefix + tok)
                    for tok in children
                }
            else:
                weights = {tok: model.next_token_prob(prefix, tok) for tok in children}
            prob *= weights[ch] / sum(weights.values())
            node = children[ch]
            prefix += ch
        dist[w] = prob
    return dist


def analytic_quotient_distribution(
    oracle: ToyOracle,
    actions: list[CompiledAction],
    *,
    use_phi: bool,
) -> dict[str, float]:
    """Exact (noise-free) induced distribution of the quotient-level sampler.

    For each string, decompose it into the fixed action sequence and multiply
    the within-action realization probabilities. With ``use_phi=True`` this must
    equal both p* and ``analytic_token_level_distribution`` exactly (the \u00a73.5
    theorem: changing the unit of speculation does not change the target).
    """
    model = oracle.model
    dist: dict[str, float] = {}
    for w in oracle.valid_strings:
        prob = 1.0
        prefix = ""
        rest = w
        for action in actions:
            match = next(s for s in action.realizations if rest.startswith(s))
            if len(action.realizations) > 1:
                if use_phi:
                    weights = {
                        s: model.continuation_prob(prefix, s) * oracle.phi(prefix + s)
                        for s in action.realizations
                    }
                else:
                    weights = {
                        s: model.continuation_prob(prefix, s) for s in action.realizations
                    }
                prob *= weights[match] / sum(weights.values())
            prefix += match
            rest = rest[len(match):]
        dist[w] = prob
    return dist


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(p || q) over the support of ``p`` where p > 0."""
    total = 0.0
    for w, pw in p.items():
        if pw <= 0.0:
            continue
        total += pw * math.log(pw / q[w])
    return total


# Backwards-compatible alias.
kl_empirical_to_ground_truth = kl_divergence


def build_oracle(
    seed: int = 0, id_granularity: str = "three_digits"
) -> tuple[ToyOracle, list[CompiledAction]]:
    """Build the toy model, exact oracle, and compiled actions in one call."""
    valid = enumerate_all_valid_strings()
    vocab = sorted(set("".join(valid)))
    model = SyntheticToyModel(vocabulary=vocab, seed=seed)
    oracle = ToyOracle(model=model, valid_strings=valid)
    actions = compile_actions(id_granularity=id_granularity)
    return oracle, actions


def run_step1_exact(
    *, seed: int = 0, id_granularity: str = "three_digits"
) -> dict[str, float]:
    """Exact KL(condition || p*) for all four conditions, with no sampling noise.

    This is the authoritative correctness check for Step 1: the two Phi
    conditions must be ~0 and equal each other, while the two no-Phi conditions
    must be strictly positive.
    """
    oracle, actions = build_oracle(seed=seed, id_granularity=id_granularity)
    conditions = {
        "token_mask_no_phi": analytic_token_level_distribution(oracle, use_phi=False),
        "token_mask_phi": analytic_token_level_distribution(oracle, use_phi=True),
        "quotient_no_phi": analytic_quotient_distribution(oracle, actions, use_phi=False),
        "quotient_phi": analytic_quotient_distribution(oracle, actions, use_phi=True),
    }
    return {name: kl_divergence(dist, oracle.p_star) for name, dist in conditions.items()}



def run_step1_experiment(
    *,
    n_samples: int = 50000,
    seed: int = 0,
    id_granularity: str = "three_digits",
    draft_mode: str = "target",
) -> dict[str, float]:
    """Run the 2x2 experiment and return KL scores for each condition.

    Conditions:
    - token_mask_no_phi
    - token_mask_phi
    - quotient_no_phi
    - quotient_phi
    """
    oracle, actions = build_oracle(seed=seed, id_granularity=id_granularity)
    valid = oracle.valid_strings

    rngs = {
        "token_mask_no_phi": random.Random(seed + 11),
        "token_mask_phi": random.Random(seed + 13),
        "quotient_no_phi": random.Random(seed + 17),
        "quotient_phi": random.Random(seed + 19),
    }

    samples: dict[str, list[str]] = {
        "token_mask_no_phi": [],
        "token_mask_phi": [],
        "quotient_no_phi": [],
        "quotient_phi": [],
    }

    for _ in range(n_samples):
        samples["token_mask_no_phi"].append(
            sample_token_level(
                oracle,
                use_phi=False,
                draft_mode=draft_mode,
                rng=rngs["token_mask_no_phi"],
            )
        )
        samples["token_mask_phi"].append(
            sample_token_level(
                oracle,
                use_phi=True,
                draft_mode=draft_mode,
                rng=rngs["token_mask_phi"],
            )
        )
        samples["quotient_no_phi"].append(
            sample_quotient_level(
                oracle,
                actions,
                use_phi=False,
                draft_mode=draft_mode,
                rng=rngs["quotient_no_phi"],
            )
        )
        samples["quotient_phi"].append(
            sample_quotient_level(
                oracle,
                actions,
                use_phi=True,
                draft_mode=draft_mode,
                rng=rngs["quotient_phi"],
            )
        )

    out: dict[str, float] = {}
    for name, drawn in samples.items():
        p_hat = empirical_distribution(drawn, valid)
        out[name] = kl_empirical_to_ground_truth(p_hat, oracle.p_star)
    return out


def _print_kl_table(title: str, kl: dict[str, float]) -> None:
    print(title)
    for name in ("token_mask_no_phi", "token_mask_phi", "quotient_no_phi", "quotient_phi"):
        print(f"  {name:<20} KL(cond || p*) = {kl[name]:.3e}")


def main() -> None:
    for gran in ("three_digits", "id_1000"):
        exact = run_step1_exact(seed=0, id_granularity=gran)
        _print_kl_table(f"[exact, id_granularity={gran}]", exact)
        gap = abs(exact["token_mask_phi"] - exact["quotient_phi"])
        print(f"  token_phi vs quotient_phi gap = {gap:.3e}  (should be ~0)\n")


if __name__ == "__main__":
    main()
