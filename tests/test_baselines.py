from test_online import FakeModel, MergeTokenizer

from gqsd.baselines import sample_token_masked
from gqsd.model import LM
from gqsd.phrase_grammar import Choice, Literal, PhraseGrammar, Slot


def test_token_masked_baseline_returns_a_valid_phrase():
    grammar = PhraseGrammar(
        name="baseline",
        segments=(Literal("a"), Slot("ending", (Choice("ab", ("b",)), Choice("ac", ("c",))))),
    )
    lm = LM(tokenizer=MergeTokenizer(), model=FakeModel(), device="cpu")

    assert sample_token_masked(lm, "", grammar.start(), 4) in {"ab", "ac"}