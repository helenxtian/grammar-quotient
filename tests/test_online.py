import random
from types import SimpleNamespace

import torch

from gqsd.model import LM
from gqsd.online import sample_online_actions, sample_online_tokens
from gqsd.phrase_grammar import Choice, Literal, PhraseGrammar, Slot


class MergeTokenizer:
    bos_token_id = 0
    eos_token_id = 5
    pad_token_id = 5

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"": [], "a": [1], "ab": [3], "ac": [1, 4]}[text]

    def decode(self, ids):
        return "".join({0: "", 1: "a", 3: "ab", 4: "c", 5: ""}[token] for token in ids)


class FakeModel:
    def __call__(self, ids, attention_mask=None):
        batch, length = ids.shape
        logits = torch.full((batch, length, 6), -4.0)
        logits[:, :, 1] = 2.0
        logits[:, :, 3] = 1.5
        logits[:, :, 4] = 1.0
        return SimpleNamespace(logits=logits)


def _lm():
    return LM(tokenizer=MergeTokenizer(), model=FakeModel(), device="cpu")


def _grammar():
    return PhraseGrammar(
        name="online_test",
        segments=(
            Literal("a"),
            Slot("ending", (Choice("merged", ("b",)), Choice("stable", ("c",)))),
        ),
    )


def test_action_verifier_batches_candidates_and_repairs_tokens():
    sample = sample_online_actions(_lm(), _grammar(), rng=random.Random(4))
    assert sample.text in {"ab", "ac"}
    assert sample.token_ids in {(3,), (1, 4)}
    assert sample.target_forward_passes == 2
    assert sample.target_rows_scored == 3
    assert sample.reclaimed_boundaries >= 1


def test_token_verifier_uses_more_target_calls_for_same_grammar():
    action = sample_online_actions(_lm(), _grammar(), rng=random.Random(7))
    token = sample_online_tokens(_lm(), _grammar(), rng=random.Random(7))
    assert token.token_ids in {(3,), (1, 4)}
    assert token.target_forward_passes > action.target_forward_passes
    assert token.target_rows_scored == token.target_forward_passes