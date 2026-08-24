"""Instrument tokenization at grammar-action boundaries."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .phrase_grammar import PhraseGrammar


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]: ...


@dataclass(frozen=True)
class TokenizationRecord:
    grammar: str
    prefix: str
    action_label: str
    realization: str
    prefix_ids: tuple[int, ...]
    joint_ids: tuple[int, ...]
    isolated_ids: tuple[int, ...]
    continuation_ids: tuple[int, ...] | None
    common_prefix_tokens: int
    rollback_tokens: int
    replacement_ids: tuple[int, ...]
    prefix_pieces: tuple[str, ...]
    joint_pieces: tuple[str, ...]
    isolated_pieces: tuple[str, ...]
    continuation_pieces: tuple[str, ...] | None
    replacement_pieces: tuple[str, ...]

    @property
    def boundary_stable(self) -> bool:
        return self.continuation_ids is not None

    @property
    def contextual_segmentation_differs(self) -> bool:
        return self.continuation_ids is not None and self.continuation_ids != self.isolated_ids

    @property
    def is_mismatch(self) -> bool:
        return not self.boundary_stable or self.contextual_segmentation_differs

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            boundary_stable=self.boundary_stable,
            contextual_segmentation_differs=self.contextual_segmentation_differs,
            is_mismatch=self.is_mismatch,
        )
        return result


@dataclass(frozen=True)
class BoundaryRepair:
    """A token-healing plan whose discarded suffix must still be uncommitted."""

    committed_prefix_ids: tuple[int, ...]
    discarded_pending_ids: tuple[int, ...]
    replacement_ids: tuple[int, ...]

    @property
    def repaired_ids(self) -> tuple[int, ...]:
        return self.committed_prefix_ids + self.replacement_ids


@dataclass(frozen=True)
class FrontierUpdate:
    """Token emission and repair metadata for one accepted text action."""

    emitted_ids: tuple[int, ...]
    discarded_pending_ids: tuple[int, ...]
    pending_ids: tuple[int, ...]
    canonical_ids: tuple[int, ...]

    @property
    def reclaimed_boundary(self) -> bool:
        return bool(self.discarded_pending_ids)


@dataclass
class TokenFrontier:
    """Keep a suffix uncommitted so the next action may retokenize its boundary."""

    tokenizer: Tokenizer
    pending_token_budget: int = 1
    text: str = ""
    committed_ids: tuple[int, ...] = ()
    pending_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.pending_token_budget < 0:
            raise ValueError("Pending token budget must be nonnegative")

    @property
    def canonical_ids(self) -> tuple[int, ...]:
        return self.committed_ids + self.pending_ids

    def append(self, realization: str) -> FrontierUpdate:
        """Accept text, retokenize canonically, and emit all but the pending suffix."""
        previous_ids = self.canonical_ids
        joint_ids = _encode(self.tokenizer, self.text + realization)
        common = 0
        for previous_id, joint_id in zip(previous_ids, joint_ids, strict=False):
            if previous_id != joint_id:
                break
            common += 1
        rollback = len(previous_ids) - common
        if common < len(self.committed_ids) or rollback > len(self.pending_ids):
            raise ValueError(
                f"Boundary repair needs {rollback} pending tokens, "
                f"only {len(self.pending_ids)} are available"
            )

        pending_count = min(self.pending_token_budget, len(joint_ids))
        new_committed = joint_ids[: len(joint_ids) - pending_count]
        new_pending = joint_ids[len(joint_ids) - pending_count :]
        emitted = new_committed[len(self.committed_ids) :]
        discarded = previous_ids[common:]

        self.text += realization
        self.committed_ids = new_committed
        self.pending_ids = new_pending
        return FrontierUpdate(
            emitted_ids=emitted,
            discarded_pending_ids=discarded,
            pending_ids=new_pending,
            canonical_ids=joint_ids,
        )

    def finalize(self) -> tuple[int, ...]:
        """Commit and return the remaining suffix at the grammar accepting state."""
        final = self.pending_ids
        self.committed_ids += final
        self.pending_ids = ()
        return final


def plan_boundary_repair(
    record: TokenizationRecord, *, pending_token_budget: int
) -> BoundaryRepair:
    """Plan canonical joint tokenization using a bounded uncommitted suffix.

    This does not permit changing tokens already consumed by a target model.
    The caller must retain ``pending_token_budget`` suffix tokens until the
    following grammar action has been tokenized and verified.
    """
    if pending_token_budget < 0:
        raise ValueError("Pending token budget must be nonnegative")
    if record.rollback_tokens > pending_token_budget:
        raise ValueError(
            f"Boundary needs {record.rollback_tokens} pending tokens, "
            f"budget is {pending_token_budget}"
        )
    split = record.common_prefix_tokens
    return BoundaryRepair(
        committed_prefix_ids=record.prefix_ids[:split],
        discarded_pending_ids=record.prefix_ids[split:],
        replacement_ids=record.replacement_ids,
    )


def _encode(tokenizer: Tokenizer, text: str) -> tuple[int, ...]:
    return tuple(tokenizer.encode(text, add_special_tokens=False))


def _pieces(tokenizer: Tokenizer, ids: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(str(piece) for piece in tokenizer.convert_ids_to_tokens(list(ids)))


def analyze_boundary(
    tokenizer: Tokenizer,
    *,
    grammar_name: str,
    prefix: str,
    action_label: str,
    realization: str,
) -> TokenizationRecord:
    """Analyze one canonical grammar action at its actual text prefix."""
    prefix_ids = _encode(tokenizer, prefix)
    joint_ids = _encode(tokenizer, prefix + realization)
    isolated_ids = _encode(tokenizer, realization)
    common_prefix_tokens = 0
    for prefix_id, joint_id in zip(prefix_ids, joint_ids, strict=False):
        if prefix_id != joint_id:
            break
        common_prefix_tokens += 1
    replacement_ids = joint_ids[common_prefix_tokens:]
    if joint_ids[: len(prefix_ids)] == prefix_ids:
        continuation_ids = joint_ids[len(prefix_ids) :]
        continuation_pieces = _pieces(tokenizer, continuation_ids)
    else:
        continuation_ids = None
        continuation_pieces = None
    return TokenizationRecord(
        grammar=grammar_name,
        prefix=prefix,
        action_label=action_label,
        realization=realization,
        prefix_ids=prefix_ids,
        joint_ids=joint_ids,
        isolated_ids=isolated_ids,
        continuation_ids=continuation_ids,
        common_prefix_tokens=common_prefix_tokens,
        rollback_tokens=len(prefix_ids) - common_prefix_tokens,
        replacement_ids=replacement_ids,
        prefix_pieces=_pieces(tokenizer, prefix_ids),
        joint_pieces=_pieces(tokenizer, joint_ids),
        isolated_pieces=_pieces(tokenizer, isolated_ids),
        continuation_pieces=continuation_pieces,
        replacement_pieces=_pieces(tokenizer, replacement_ids),
    )


def scan_grammar(
    tokenizer: Tokenizer, grammar: PhraseGrammar
) -> list[TokenizationRecord]:
    """Analyze every realization at every reachable state of a finite grammar."""
    records: list[TokenizationRecord] = []
    frontier = [grammar.start()]
    seen = set()
    while frontier:
        state = frontier.pop()
        if state in seen:
            continue
        seen.add(state)
        if state.is_accepting():
            continue
        for action in state.actions():
            for realization in action.realizations:
                records.append(
                    analyze_boundary(
                        tokenizer,
                        grammar_name=grammar.name,
                        prefix=state.text,
                        action_label=action.label,
                        realization=realization,
                    )
                )
                frontier.append(state.advance(action, realization))
    return records


def mismatch_summary(
    records: list[TokenizationRecord], *, pending_token_budget: int = 1
) -> dict[str, int]:
    """Return stable counters suitable for benchmark output and regression tests."""
    if pending_token_budget < 0:
        raise ValueError("Pending token budget must be nonnegative")
    return {
        "records": len(records),
        "mismatches": sum(record.is_mismatch for record in records),
        "boundary_crossings": sum(not record.boundary_stable for record in records),
        "contextual_segmentations": sum(
            record.contextual_segmentation_differs for record in records
        ),
        "max_rollback_tokens": max((record.rollback_tokens for record in records), default=0),
        "reclaimable_mismatches": sum(
            record.is_mismatch and record.rollback_tokens <= pending_token_budget
            for record in records
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grammars", nargs="+", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--revision")
    parser.add_argument("--coalesce-literals", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--pending-token-budget", type=int, default=1)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    tokenizer_file = Path(
        cached_file(
            args.model,
            "tokenizer.json",
            revision=args.revision,
            local_files_only=args.local_files_only,
        )
    )
    output: dict[str, Any] = {
        "model": args.model,
        "revision": tokenizer_file.parent.name,
        "pending_token_budget": args.pending_token_budget,
        "grammars": {},
    }
    for path in args.grammars:
        grammar = PhraseGrammar.load(path)
        if args.coalesce_literals:
            grammar = grammar.coalesce_literals()
        records = scan_grammar(tokenizer, grammar)
        mismatches = [record for record in records if record.is_mismatch]
        output["grammars"][path.name] = {
            "support_size": len(grammar.enumerate()),
            "summary": mismatch_summary(
                records, pending_token_budget=args.pending_token_budget
            ),
            "examples": [record.to_dict() for record in mismatches[: args.examples]],
        }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())