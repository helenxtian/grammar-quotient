"""Thin wrapper around a small HF causal LM so we control logits and the cache.

Phase 0 deliverable: a base model we can get next-token logits from cheaply.
Everything downstream (action masses, future-validity rollouts, draft/target
verification) is built on the primitives exposed here.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import Error as CopyError
from copy import deepcopy
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
    target_forward_passes: int = 0

    @classmethod
    def load(
        cls,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        *,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> LM:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            dtype=torch.float32 if device == "cpu" else torch.float16,
        ).to(device)
        model.eval()
        return cls(tokenizer=tok, model=model, device=device)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def initial_context_ids(self) -> list[int]:
        """Return a single model token that can predict the first output token."""
        token_id = self.tokenizer.bos_token_id
        if token_id is None:
            token_id = self.tokenizer.eos_token_id
        if token_id is None:
            raise ValueError("Tokenizer needs a BOS or EOS token to score an empty prefix")
        return [token_id]

    def reset_counters(self) -> None:
        self.target_forward_passes = 0

    @torch.no_grad()
    def sequence_logprob_with_prefix_cache(
        self, prefix_ids: list[int], cont_ids: list[int]
    ) -> float:
        """Score a continuation while reusing one forward pass for its prefix."""
        if not cont_ids:
            return 0.0
        context_ids = prefix_ids or self.initial_context_ids()
        prefix_tensor = torch.tensor([context_ids], device=self.device)
        prefix_output = self._forward(prefix_tensor, attention_mask=torch.ones_like(prefix_tensor))
        past = prefix_output.past_key_values
        if past is None:
            return self.sequence_logprob(prefix_ids, cont_ids)
        total = 0.0
        logits = prefix_output.logits[0, -1]
        total += torch.log_softmax(logits.float(), dim=-1)[cont_ids[0]].item()
        for token_id, previous_id in zip(cont_ids[1:], cont_ids, strict=False):
            next_input = torch.tensor([[previous_id]], device=self.device)
            output = self._forward(next_input, past_key_values=past, use_cache=True)
            past = output.past_key_values
            logits = output.logits[0, -1]
            total += torch.log_softmax(logits.float(), dim=-1)[token_id].item()
        return total

    @torch.no_grad()
    def batch_sequence_logprobs_with_prefix_cache(
        self, prefix_ids: list[int], continuations: list[list[int]]
    ) -> list[float] | None:
        """Score branches sharing a prefix, reusing the prefix forward pass.

        Returns ``None`` when the underlying model does not expose a usable
        cache, allowing callers to retain their regular batched scorer.
        """
        if not continuations:
            return []
        if any(not continuation for continuation in continuations):
            raise ValueError("Cached continuations must not be empty")
        context_ids = prefix_ids or self.initial_context_ids()
        prefix_tensor = torch.tensor([context_ids], device=self.device)
        try:
            prefix_output = self._forward(
                prefix_tensor,
                attention_mask=torch.ones_like(prefix_tensor),
                use_cache=True,
            )
            prefix_past = prefix_output.past_key_values
        except (AttributeError, TypeError):
            return None
        if prefix_past is None:
            return None

        prefix_logprobs = torch.log_softmax(prefix_output.logits[0, -1].float(), dim=-1)
        scores: list[float] = []
        for continuation in continuations:
            try:
                past = deepcopy(prefix_past)
            except (CopyError, TypeError, RuntimeError):
                return None
            total = prefix_logprobs[continuation[0]].item()
            for offset, token_id in enumerate(continuation[1:], start=1):
                previous_id = continuation[offset - 1]
                input_tensor = torch.tensor([[previous_id]], device=self.device)
                attention_mask = torch.ones(
                    (1, len(context_ids) + offset), device=self.device, dtype=torch.long
                )
                try:
                    output = self._forward(
                        input_tensor,
                        attention_mask=attention_mask,
                        past_key_values=past,
                        use_cache=True,
                    )
                except (AttributeError, TypeError, RuntimeError):
                    return None
                past = output.past_key_values
                if past is None:
                    return None
                logits = output.logits[0, -1].float()
                total += torch.log_softmax(logits, dim=-1)[token_id].item()
            scores.append(total)
        return scores

    def _forward(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        self.target_forward_passes += 1
        if attention_mask is None:
            return self.model(ids, **kwargs)
        return self.model(ids, attention_mask=attention_mask, **kwargs)

    @torch.no_grad()
    def next_token_logprobs(self, input_ids: list[int]) -> torch.Tensor:
        """Log p(. | input_ids) over the full vocab, shape [vocab]."""
        context_ids = input_ids or self.initial_context_ids()
        ids = torch.tensor([context_ids], device=self.device)
        logits = self._forward(ids).logits[0, -1]  # [vocab]
        return torch.log_softmax(logits.float(), dim=-1)

    @torch.no_grad()
    def sequence_logprob(self, prefix_ids: list[int], cont_ids: list[int]) -> float:
        """log p(cont_ids | prefix_ids), summed over the continuation tokens.

        This is the workhorse for an action mass Sigma_{s in C} p(s | u): score
        each concrete token realization s of an action and sum in prob space.
        """
        if not cont_ids:
            return 0.0
        context_ids = prefix_ids or self.initial_context_ids()
        ids = torch.tensor([context_ids + cont_ids], device=self.device)
        logits = self._forward(ids).logits[0]  # [seq, vocab]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        total = 0.0
        start = len(context_ids) - 1  # logits at position t predict token t+1
        for k, tok in enumerate(cont_ids):
            total += logprobs[start + k, tok].item()
        return total

    def text_logprob(self, prefix: str, continuation: str) -> float:
        """Return log p(continuation | prefix) under canonical joint tokenization.

        A text boundary is scoreable as a token continuation only when encoding
        ``prefix + continuation`` preserves the tokenization of ``prefix``.
        Callers must choose an earlier grammar boundary when a tokenizer merge
        crosses the requested boundary.
        """
        prefix_ids = self.encode(prefix)
        joint_ids = self.encode(prefix + continuation)
        if joint_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError("Prefix/continuation boundary is not tokenization-stable")
        return self.sequence_logprob(prefix_ids, joint_ids[len(prefix_ids) :])

    def batch_text_logprobs(
        self, prefix: str, continuations: Iterable[str], *, batch_size: int = 8
    ) -> list[float]:
        """Score finite grammar realizations against the same text prefix."""
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        examples: list[tuple[list[int], list[int]]] = []
        prefix_ids = self.encode(prefix)
        for continuation in continuations:
            joint_ids = self.encode(prefix + continuation)
            if joint_ids[: len(prefix_ids)] != prefix_ids:
                raise ValueError("Prefix/continuation boundary is not tokenization-stable")
            examples.append((prefix_ids, joint_ids[len(prefix_ids) :]))
        scores: list[float] = []
        for start in range(0, len(examples), batch_size):
            scores.extend(
                self.batch_sequence_logprobs(examples[start : start + batch_size])
            )
        return scores

    @torch.no_grad()
    def batch_sequence_logprobs(
        self, examples: list[tuple[list[int], list[int]]]
    ) -> list[float]:
        if not examples:
            return []
        rows: list[list[int]] = []
        starts: list[int] = []
        continuation_lengths: list[int] = []
        for prefix_ids, continuation_ids in examples:
            context_ids = prefix_ids or self.initial_context_ids()
            rows.append(context_ids + continuation_ids)
            starts.append(len(context_ids) - 1)
            continuation_lengths.append(len(continuation_ids))

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.bos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer needs a padding, EOS, or BOS token for batched scoring")

        max_length = max(len(row) for row in rows)
        padded = [row + [pad_token_id] * (max_length - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (max_length - len(row)) for row in rows]
        ids = torch.tensor(padded, device=self.device)
        attention_mask = torch.tensor(masks, device=self.device)
        logits = self._forward(ids, attention_mask=attention_mask).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)

        totals: list[float] = []
        for row_index, ((_, continuation_ids), start, length) in enumerate(
            zip(examples, starts, continuation_lengths, strict=True)
        ):
            total = 0.0
            for offset, token_id in enumerate(continuation_ids[:length]):
                total += logprobs[row_index, start + offset, token_id].item()
            totals.append(total)
        return totals
