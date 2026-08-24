"""Grammar substrate: states, macro actions C(g), and their token realizations.

Phase 0 decision (see project plan): start with a hand-written, fixed-schema
JSON-object grammar -- NOT a general CFG engine. In this regime every action is
deterministic or finite, so action masses Sigma_{s in C} p(s | u) are computable
exactly, which is what the correctness experiments need.

The base state is intentionally grammar-agnostic. Concrete grammars expose a
finite set of competing actions and advance immutably after a realization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Action label cannot be empty")
        if self.kind is ActionKind.OPEN and self.realizations:
            raise ValueError("Open actions cannot enumerate realizations")
        if self.kind is not ActionKind.OPEN and not self.realizations:
            raise ValueError("Finite and deterministic actions need realizations")
        if self.kind is ActionKind.DETERMINISTIC and len(self.realizations) != 1:
            raise ValueError("Deterministic actions need exactly one realization")
        if len(set(self.realizations)) != len(self.realizations):
            raise ValueError("Action realizations must be unique")


class GrammarState(ABC):
    """Opaque immutable grammar state that owns the next action partition."""

    @abstractmethod
    def actions(self) -> list[Action]:
        """Return C(g): the macro actions permitted from this state."""

    @abstractmethod
    def advance(self, action: Action, realization: str) -> GrammarState:
        """Return g' after committing to `action` realized as `realization`."""

    @abstractmethod
    def is_accepting(self) -> bool:
        """True if the string built so far is a complete valid member of L."""
