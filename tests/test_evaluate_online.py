from types import SimpleNamespace

import pytest
import torch

from gqsd.evaluate_online import run_online_comparison
from gqsd.model import LM
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
        name="online_eval_test",
        segments=(
            Literal("a"),
            Slot("ending", (Choice("merged", ("b",)), Choice("stable", ("c",)))),
        ),
    )


def test_online_comparison_measures_actual_call_advantage():
    result = run_online_comparison(_lm(), _grammar(), samples=20, seed=3)
    assert result["action"]["canonical_tokenizations"] == 20
    assert result["token"]["canonical_tokenizations"] == 20
    assert result["action"]["target_forward_passes"] < result["token"]["target_forward_passes"]
    assert result["action"]["reclaimed_boundaries"] > 0
    assert result["token"]["reclaimed_boundaries"] > 0
    assert result["comparison"]["target_call_ratio_token_over_action"] > 1.0
    assert result["comparison"]["target_call_reduction"] > 0.0
    assert result["comparison"]["output_tokens_per_call_ratio_action_over_token"] > 1.0


def test_online_comparison_requires_samples():
    with pytest.raises(ValueError, match="must be positive"):
        run_online_comparison(_lm(), _grammar(), samples=0, seed=3)