"""Phase 0 smoke tests: the package imports and the scaffolding is wired.

These deliberately do NOT load a model (keeps CI/dev fast and offline). Model-
backed tests arrive in Phase 1 alongside the oracle.
"""

import json
from pathlib import Path

import gqsd
from gqsd.grammar import Action, ActionKind


def test_version():
    assert gqsd.__version__ == "0.0.1"


def test_action_dataclass():
    a = Action(kind=ActionKind.FINITE, label="status", realizations=("open", "closed"))
    assert a.kind is ActionKind.FINITE
    assert "open" in a.realizations


def test_toy_grammar_spec_loads():
    spec = json.loads((Path(__file__).parents[1] / "grammars" / "toy_record.json").read_text())
    assert spec["field_order"] == ["status", "priority", "approved"]
    assert spec["fields"]["status"]["enum"] == ["open", "closed", "pending"]
