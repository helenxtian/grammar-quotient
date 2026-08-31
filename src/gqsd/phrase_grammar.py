"""Finite phrase grammars for natural-looking, exactly enumerable languages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from .grammar import Action, ActionKind, GrammarState


@dataclass(frozen=True)
class Literal:
    text: str


@dataclass(frozen=True)
class Choice:
    label: str
    realizations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.label or not self.realizations:
            raise ValueError("A choice needs a label and at least one realization")
        if len(set(self.realizations)) != len(self.realizations):
            raise ValueError("Choice realizations must be unique")


@dataclass(frozen=True)
class Slot:
    name: str
    choices: tuple[Choice, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.choices:
            raise ValueError("A slot needs a name and at least one choice")
        labels = [choice.label for choice in self.choices]
        if len(set(labels)) != len(labels):
            raise ValueError(f"Choice labels must be unique within slot {self.name!r}")
        realizations = [text for choice in self.choices for text in choice.realizations]
        if len(set(realizations)) != len(realizations):
            raise ValueError(f"Choices in slot {self.name!r} must be disjoint")


@dataclass(frozen=True)
class OpenSpan:
    name: str
    stop: str
    max_tokens: int = 32

    def __post_init__(self) -> None:
        if not self.name or not self.stop:
            raise ValueError("An open span needs a name and stop marker")
        if self.max_tokens <= 0:
            raise ValueError("An open span needs a positive token budget")


Segment = Literal | Slot | OpenSpan


@dataclass(frozen=True)
class PhraseGrammar:
    name: str
    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.segments:
            raise ValueError("A phrase grammar needs a name and segments")

    def start(self) -> PhraseState:
        return PhraseState(grammar=self)

    def enumerate(self) -> list[str]:
        if any(isinstance(segment, OpenSpan) for segment in self.segments):
            raise ValueError("Open-span grammars do not have a finite enumeration")
        alternatives: list[tuple[str, ...]] = []
        for segment in self.segments:
            if isinstance(segment, Literal):
                alternatives.append((segment.text,))
            elif isinstance(segment, Slot):
                alternatives.append(
                    tuple(text for choice in segment.choices for text in choice.realizations)
                )
        return ["".join(parts) for parts in product(*alternatives)]

    def coalesce_literals(self) -> PhraseGrammar:
        """Fold each deterministic literal into the following action realizations.

        BPE tokenizers commonly merge trailing whitespace or punctuation with
        the first token of the next phrase. Moving that literal into the action
        keeps the generated language unchanged while placing the grammar
        boundary before the tokenizer merge.
        """
        segments: list[Segment] = []
        pending_literal = ""
        for segment in self.segments:
            if isinstance(segment, Literal):
                pending_literal += segment.text
                continue
            if isinstance(segment, OpenSpan):
                raise TypeError("Cannot coalesce literals around an open span")
            if pending_literal:
                segment = Slot(
                    name=segment.name,
                    choices=tuple(
                        Choice(
                            label=choice.label,
                            realizations=tuple(
                                pending_literal + realization
                                for realization in choice.realizations
                            ),
                        )
                        for choice in segment.choices
                    ),
                )
                pending_literal = ""
            segments.append(segment)
        if pending_literal:
            segments.append(Literal(pending_literal))
        return PhraseGrammar(name=self.name, segments=tuple(segments))

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> PhraseGrammar:
        segments: list[Segment] = []
        for raw_segment in spec["segments"]:
            if "literal" in raw_segment:
                segments.append(Literal(text=str(raw_segment["literal"])))
                continue
            choices = tuple(
                Choice(
                    label=str(raw_choice["label"]),
                    realizations=tuple(str(text) for text in raw_choice["realizations"]),
                )
                for raw_choice in raw_segment["choices"]
            )
            segments.append(Slot(name=str(raw_segment["slot"]), choices=choices))
        return cls(name=str(spec["name"]), segments=tuple(segments))

    @classmethod
    def load(cls, path: str | Path) -> PhraseGrammar:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass(frozen=True)
class PhraseState(GrammarState):
    grammar: PhraseGrammar
    segment_index: int = 0
    text: str = ""

    def actions(self) -> list[Action]:
        if self.is_accepting():
            return []
        segment = self.grammar.segments[self.segment_index]
        if isinstance(segment, Literal):
            return [
                Action(
                    kind=ActionKind.DETERMINISTIC,
                    label=f"literal:{self.segment_index}",
                    realizations=(segment.text,),
                )
            ]
        if isinstance(segment, OpenSpan):
            return [
                Action(
                    kind=ActionKind.OPEN,
                    label=f"{segment.name}:{self.segment_index}",
                )
            ]
        return [
            Action(
                kind=(
                    ActionKind.DETERMINISTIC
                    if len(choice.realizations) == 1
                    else ActionKind.FINITE
                ),
                label=f"{segment.name}:{choice.label}",
                realizations=choice.realizations,
            )
            for choice in segment.choices
        ]

    def advance(self, action: Action, realization: str) -> PhraseState:
        if action not in self.actions():
            raise ValueError(f"Action {action.label!r} is not available from this state")
        if action.kind is not ActionKind.OPEN and realization not in action.realizations:
            raise ValueError(f"Realization {realization!r} does not belong to {action.label!r}")
        return PhraseState(
            grammar=self.grammar,
            segment_index=self.segment_index + 1,
            text=self.text + realization,
        )

    def is_accepting(self) -> bool:
        return self.segment_index == len(self.grammar.segments)


def enumerate_from_actions(start: PhraseState) -> list[str]:
    """Enumerate through the public state API, useful for contract validation."""
    outputs: list[str] = []
    frontier: list[PhraseState] = [start]
    while frontier:
        state = frontier.pop()
        if state.is_accepting():
            outputs.append(state.text)
            continue
        for action in state.actions():
            frontier.extend(state.advance(action, text) for text in action.realizations)
    return sorted(outputs)