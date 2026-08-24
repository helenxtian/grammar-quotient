from types import SimpleNamespace

import pytest
import torch

from gqsd.model import LM


class FakeTokenizer:
    bos_token_id = 0
    eos_token_id = 4
    pad_token_id = 4

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == "ab":
            return [3]
        return {"": [], "a": [1], "b": [2]}[text]

    def decode(self, ids):
        return "".join({0: "", 1: "a", 2: "b", 3: "ab", 4: ""}[token_id] for token_id in ids)


class FakeModel:
    def __call__(self, ids, attention_mask=None):
        batch, length = ids.shape
        logits = torch.full((batch, length, 5), -20.0)
        logits[:, :, 1] = 2.0
        logits[:, :, 2] = 1.0
        logits[:, :, 3] = 0.0
        return SimpleNamespace(logits=logits)


def _lm():
    return LM(tokenizer=FakeTokenizer(), model=FakeModel(), device="cpu")


def test_empty_prefix_uses_initial_context_to_score_first_token():
    lm = _lm()
    expected = torch.log_softmax(torch.tensor([-20.0, 2.0, 1.0, 0.0, -20.0]), dim=-1)[1]
    assert lm.sequence_logprob([], [1]) == pytest.approx(expected.item())


def test_text_logprob_rejects_cross_boundary_token_merge():
    with pytest.raises(ValueError, match="not tokenization-stable"):
        _lm().text_logprob("a", "b")


def test_batch_text_logprobs_scores_stable_realizations():
    scores = _lm().batch_text_logprobs("", ["a", "b"], batch_size=1)
    assert len(scores) == 2
    assert scores[0] > scores[1]
    assert scores == pytest.approx(
        [_lm().text_logprob("", "a"), _lm().text_logprob("", "b")]
    )


def test_batch_text_logprobs_rejects_nonpositive_batch_size():
    with pytest.raises(ValueError, match="must be positive"):
        _lm().batch_text_logprobs("", ["a"], batch_size=0)


def test_model_counts_actual_forward_calls():
    lm = _lm()
    lm.batch_text_logprobs("", ["a", "b"], batch_size=1)
    assert lm.target_forward_passes == 2
    lm.next_token_logprobs([])
    assert lm.target_forward_passes == 3
    lm.reset_counters()
    assert lm.target_forward_passes == 0