"""Grammar substrate: states, macro actions C(g), and their token realizations.

Phase 0 decision (see project plan): start with a hand-written, fixed-schema
JSON-object grammar -- NOT a general CFG engine. In this regime every action is
deterministic or finite, so action masses Sigma_{s in C} p(s | u) are computable
exactly, which is what the correctness experiments need.

This module is a stub: the interfaces below define the contract the rest of the
system codes against. Phase 2 fills in the JSON-object implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionKind(Enum):
    DETERMINISTIC = "deterministic"  # canonical realization (field name, delimiter)
    FINITE = "finite"  # finite alias / enum / id / field set
    OPEN = "open"  # open-ended span -> fall back to token decoding


@dataclass(frozen=True)
class Action:
    """A grammar-level macro action C: an equivalence class of continuations.

    `realizations` lists the concrete token strings s in C (as decoded text);
    the model wrapper turns each into token ids for scoring. For OPEN actions
    `realizations` is empty and decoding falls back to the token level.
    """

    kind: ActionKind
    label: str
    realizations: tuple[str, ...] = ()


class GrammarState:
    """Opaque grammar state g. Knows which actions are permitted next.

    TODO(phase-2): implement for the fixed-schema JSON object grammar.
    """

    def actions(self) -> list[Action]:
        """Return C(g): the macro actions permitted from this state."""
        raise NotImplementedError("phase 2: JSON-object grammar")

    def advance(self, action: Action, realization: str) -> "GrammarState":
        """Return g' after committing to `action` realized as `realization`."""
        raise NotImplementedError("phase 2: JSON-object grammar")

    def is_accepting(self) -> bool:
        """True if the string built so far is a complete valid member of L."""
        raise NotImplementedError("phase 2: JSON-object grammar")
