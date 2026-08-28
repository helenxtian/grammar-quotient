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


def test_generate_actions_preserves_prompt_in_token_frontier():
    grammar = PhraseGrammar(name="generation", segments=(Literal("a"),))

    class PromptTokenizer(MergeTokenizer):
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return {"": [], "p": [2], "pa": [2, 1]}[text]

    result = generate_actions(
        LM(tokenizer=PromptTokenizer(), model=FakeModel(), device="cpu"),
        grammar,
        prompt="p",
        rng=random.Random(4),
    )

    assert result.text == "a"