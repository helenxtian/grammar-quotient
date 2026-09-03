import random
from types import SimpleNamespace

import torch
from test_online import FakeModel, MergeTokenizer

from gqsd.generate import choose_cache_policy, generate_actions
from gqsd.model import LM
from gqsd.phrase_grammar import Choice, Literal, OpenSpan, PhraseGrammar, Slot


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


class CacheFakeModel(FakeModel):
    def __init__(self):
        self.calls = 0

    def __call__(self, ids, attention_mask=None, past_key_values=None, use_cache=False):
        self.calls += 1
        output = super().__call__(ids, attention_mask=attention_mask)
        output.past_key_values = (self.calls,)
        return output


def test_generate_actions_uses_prefix_cache_when_available():
    model = CacheFakeModel()
    grammar = PhraseGrammar(name="generation", segments=(Literal("a"),))

    result = generate_actions(
        LM(tokenizer=MergeTokenizer(), model=model, device="cpu"),
        grammar,
        use_prefix_cache=True,
    )

    assert result.text == "a"
    assert model.calls == 2


def test_adaptive_cache_policy_requires_deep_branches():
    assert not choose_cache_policy(3, [1, 2, 2])
    assert choose_cache_policy(20, [4, 4])


class OpenTokenizer:
    bos_token_id = 0
    eos_token_id = 4
    pad_token_id = 4

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [0] if text == "" else [{"x": 1, ".": 2, "x.": 3}[text]]

    def decode(self, ids):
        return "".join({0: "", 1: "x", 2: ".", 3: "x."}[token_id] for token_id in ids)


class OpenModel:
    def __call__(self, ids, attention_mask=None, **kwargs):
        batch, length = ids.shape
        logits = torch.full((batch, length, 5), -20.0)
        logits[:, :, 1] = 3.0 if ids[0, -1].item() == 0 else -20.0
        logits[:, :, 2] = 3.0 if ids[0, -1].item() == 1 else -20.0
        return SimpleNamespace(logits=logits)


def test_generate_actions_supports_bounded_open_span():
    grammar = PhraseGrammar(
        name="open",
        segments=(OpenSpan(name="body", stop=".", max_tokens=2),),
    )

    result = generate_actions(LM(tokenizer=OpenTokenizer(), model=OpenModel(), device="cpu"), grammar)

    assert result.text == "x."
    assert result.actions[0].realization == "x"