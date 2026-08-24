import pytest

from gqsd.phrase_grammar import Choice, Literal, PhraseGrammar, Slot
from gqsd.tokenizer_analysis import (
    TokenFrontier,
    analyze_boundary,
    mismatch_summary,
    plan_boundary_repair,
    scan_grammar,
)


class MergeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        ids = []
        index = 0
        while index < len(text):
            if text[index : index + 2] == "ab":
                ids.append(3)
                index += 2
            else:
                ids.append({"a": 1, "b": 2, "c": 4}[text[index]])
                index += 1
        return ids

    def convert_ids_to_tokens(self, ids):
        return [{1: "a", 2: "b", 3: "ab", 4: "c"}[token_id] for token_id in ids]


def _grammar():
    return PhraseGrammar(
        name="merge_test",
        segments=(
            Literal("a"),
            Slot("choice", (Choice("merged", ("b",)), Choice("stable", ("c",)))),
        ),
    )


def test_boundary_analysis_detects_cross_action_token_merge():
    record = analyze_boundary(
        MergeTokenizer(),
        grammar_name="merge_test",
        prefix="a",
        action_label="choice:merged",
        realization="b",
    )
    assert record.boundary_stable is False
    assert record.is_mismatch is True
    assert record.prefix_ids == (1,)
    assert record.joint_ids == (3,)
    assert record.continuation_ids is None
    assert record.rollback_tokens == 1
    assert record.replacement_ids == (3,)


def test_scanner_visits_stable_and_crossing_realizations():
    records = scan_grammar(MergeTokenizer(), _grammar())
    summary = mismatch_summary(records)
    assert summary == {
        "records": 3,
        "mismatches": 1,
        "boundary_crossings": 1,
        "contextual_segmentations": 0,
        "max_rollback_tokens": 1,
        "reclaimable_mismatches": 1,
    }
    stable = next(record for record in records if record.realization == "c")
    assert stable.boundary_stable is True
    assert stable.continuation_ids == (4,)


def test_record_serializes_derived_mismatch_fields():
    record = scan_grammar(MergeTokenizer(), _grammar())[1]
    data = record.to_dict()
    assert data["grammar"] == "merge_test"
    assert isinstance(data["is_mismatch"], bool)


def test_coalesced_literal_removes_the_crossing_boundary():
    records = scan_grammar(MergeTokenizer(), _grammar().coalesce_literals())
    assert mismatch_summary(records)["boundary_crossings"] == 0


def test_boundary_repair_reconstructs_joint_tokens_from_pending_suffix():
    record = analyze_boundary(
        MergeTokenizer(),
        grammar_name="merge_test",
        prefix="a",
        action_label="choice:merged",
        realization="b",
    )
    repair = plan_boundary_repair(record, pending_token_budget=1)
    assert repair.discarded_pending_ids == (1,)
    assert repair.replacement_ids == (3,)
    assert repair.repaired_ids == record.joint_ids


def test_boundary_repair_rejects_insufficient_pending_budget():
    record = analyze_boundary(
        MergeTokenizer(),
        grammar_name="merge_test",
        prefix="a",
        action_label="choice:merged",
        realization="b",
    )
    with pytest.raises(ValueError, match="Boundary needs 1 pending tokens"):
        plan_boundary_repair(record, pending_token_budget=0)


def test_summary_rejects_negative_pending_budget():
    with pytest.raises(ValueError, match="must be nonnegative"):
        mismatch_summary([], pending_token_budget=-1)


def test_pending_frontier_reclaims_merge_and_emits_canonical_tokens():
    frontier = TokenFrontier(MergeTokenizer(), pending_token_budget=1)
    first = frontier.append("a")
    assert first.emitted_ids == ()
    assert first.pending_ids == (1,)

    second = frontier.append("b")
    assert second.reclaimed_boundary is True
    assert second.discarded_pending_ids == (1,)
    assert second.canonical_ids == (3,)
    assert frontier.finalize() == (3,)
    assert frontier.canonical_ids == (3,)


def test_pending_frontier_rejects_merge_after_token_was_committed():
    frontier = TokenFrontier(MergeTokenizer(), pending_token_budget=0)
    frontier.append("a")
    with pytest.raises(ValueError, match="only 0 are available"):
        frontier.append("b")