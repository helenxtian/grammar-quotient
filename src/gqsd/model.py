"""Thin wrapper around a small HF causal LM so we control logits and the cache.

Phase 0 deliverable: a base model we can get next-token logits from cheaply.
Everything downstream (action masses, future-validity rollouts, draft/target
verification) is built on the primitives exposed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Small, logit-accessible defaults. Override via LM.load(model_id=...).
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"


@dataclass
class LM:
    """Minimal causal-LM interface: tokenize, and score next-token logits.

    Intentionally tiny — the research code should depend only on these few
    primitives, not on the full generate() stack, so the quotient layer stays
    explicit.
    """

    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: str

    @classmethod
    def load(cls, model_id: str = DEFAULT_MODEL, device: str | None = None) -> "LM":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        ).to(device)
        model.eval()
        return cls(tokenizer=tok, model=model, device=device)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    @torch.no_grad()
    def next_token_logprobs(self, input_ids: list[int]) -> torch.Tensor:
        """Log p(. | input_ids) over the full vocab, shape [vocab]."""
        ids = torch.tensor([input_ids], device=self.device)
        logits = self.model(ids).logits[0, -1]  # [vocab]
        return torch.log_softmax(logits.float(), dim=-1)

    @torch.no_grad()
    def sequence_logprob(self, prefix_ids: list[int], cont_ids: list[int]) -> float:
        """log p(cont_ids | prefix_ids), summed over the continuation tokens.

        This is the workhorse for an action mass Sigma_{s in C} p(s | u): score
        each concrete token realization s of an action and sum in prob space.
        """
        if not cont_ids:
            return 0.0
        ids = torch.tensor([prefix_ids + cont_ids], device=self.device)
        logits = self.model(ids).logits[0]  # [seq, vocab]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        total = 0.0
        start = len(prefix_ids) - 1  # logits at position t predict token t+1
        for k, tok in enumerate(cont_ids):
            total += logprobs[start + k, tok].item()
        return total
