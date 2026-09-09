from test_online import FakeModel, MergeTokenizer

from gqsd.cli import generate_one
from gqsd.model import LM
from gqsd.oracle import FiniteGrammarOracle
from gqsd.phrase_grammar import Choice, Literal, PhraseGrammar, Slot


def _setup():
    grammar = PhraseGrammar(
        name="cli",
        segments=(Literal("a"), Slot("ending", (Choice("b", ("b",)),))),
    )
    lm = LM(tokenizer=MergeTokenizer(), model=FakeModel(), device="cpu")
    return grammar, lm


def test_generate_one_reports_grammar_action_result():
    grammar, lm = _setup()
    oracle = FiniteGrammarOracle.from_lm(grammar, lm)

    result = generate_one(
        lm,
        grammar,
        mode="grammar_actions",
        prompt="",
        seed=4,
        max_tokens=4,
        cache_policy="padded",
        beam_size=2,
        oracle=oracle,
    )

    assert result["text"] == "ab"
    assert result["oracle_valid"] is True
    assert result["actions"]
    assert result["target_forward_passes"] > 0


def test_generate_one_supports_unconstrained_mode():
    grammar, lm = _setup()

    result = generate_one(
        lm,
        grammar,
        mode="unconstrained",
        prompt="",
        seed=4,
        max_tokens=2,
        cache_policy="padded",
        beam_size=2,
    )

    assert result["mode"] == "unconstrained"
    assert "actions" not in result
    assert isinstance(result["text"], str)
