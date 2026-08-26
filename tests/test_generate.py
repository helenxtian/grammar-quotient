import random

from test_online import FakeModel, MergeTokenizer

from gqsd.generate import generate_actions
from gqsd.model import LM
from gqsd.phrase_grammar import Choice, Literal, PhraseGrammar, Slot


def test_generate_actions_returns_trace_and_canonical_tokens():
    grammar = PhraseGrammar(
        name="generation",
        segments=(Literal("a"), Slot("ending", (Choice("ab", ("b",)), Choice("ac", ("c",))))),
    )
    result = generate_actions(
        LM(tokenizer=MergeTokenizer(), model=FakeModel(), device="cpu"),
        grammar,
        rng=random.Random(4),
    )

    assert result.text == "ab"
    assert result.token_ids == (3,)
    assert len(result.actions) == 2
    assert result.reclaimed_boundaries == 1