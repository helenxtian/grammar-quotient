from pathlib import Path

import pytest

from gqsd.phrase_grammar import (
    Choice,
    Literal,
    PhraseGrammar,
    Slot,
    enumerate_from_actions,
)


def _report_grammar():
    return PhraseGrammar(
        name="report",
        segments=(
            Literal("Assessment: "),
            Slot(
                "finding",
                (
                    Choice("stable", ("the service remained stable",)),
                    Choice("degraded", ("latency increased", "response times rose")),
                ),
            ),
            Slot(
                "recommendation",
                (
                    Choice("none", (".",)),
                    Choice("monitor", ("; continued monitoring is recommended.",)),
                ),
            ),
        ),
    )


def test_slot_exposes_competing_grammar_actions():
    state = _report_grammar().start()
    literal = state.actions()[0]
    state = state.advance(literal, literal.realizations[0])
    assert [action.label for action in state.actions()] == [
        "finding:stable",
        "finding:degraded",
    ]
    assert state.actions()[1].realizations == ("latency increased", "response times rose")


def test_state_enumeration_matches_cartesian_language():
    grammar = _report_grammar()
    direct = sorted(grammar.enumerate())
    through_actions = enumerate_from_actions(grammar.start())
    assert direct == through_actions
    assert len(direct) == 6


def test_state_rejects_wrong_action_or_realization():
    state = _report_grammar().start()
    action = state.actions()[0]
    with pytest.raises(ValueError, match="does not belong"):
        state.advance(action, "wrong")


def test_slot_choices_must_partition_realizations():
    with pytest.raises(ValueError, match="must be disjoint"):
        Slot("duplicate", (Choice("a", ("same",)), Choice("b", ("same",))))


@pytest.mark.parametrize(
    ("filename", "expected_size"),
    [
        ("report_phrases.json", 432),
        ("dialogue_phrases.json", 108),
        ("code_docstring_phrases.json", 576),
    ],
)
def test_representative_grammar_fixture(filename, expected_size):
    path = Path(__file__).parents[1] / "grammars" / filename
    grammar = PhraseGrammar.load(path)
    outputs = grammar.enumerate()
    assert len(outputs) == expected_size
    assert len(set(outputs)) == expected_size
    assert sorted(outputs) == enumerate_from_actions(grammar.start())

    state = grammar.start()
    max_competing_actions = 0
    while not state.is_accepting():
        actions = state.actions()
        max_competing_actions = max(max_competing_actions, len(actions))
        state = state.advance(actions[0], actions[0].realizations[0])
    assert max_competing_actions >= 2