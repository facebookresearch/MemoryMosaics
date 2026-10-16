# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Autoregressive generation with a KV cache, for every model family.

Each step feeds the model the tokens it has not seen: the whole prompt prefix first (a
prefill), then one token at a time. Under attn_bias_type "doc_causal" a prompt may hold
several documents separated by EOS, and generation may cross EOS into a new document:
the model is told after every step (model.end_documents), so positions, memories and
recurrent states follow document boundaries exactly as in training.
"""

import logging
from collections.abc import Iterator

import torch

from model.common.mask import tokens_to_seq_lens

logger = logging.getLogger()


def _sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    topk_probs, topk_ids = torch.topk(probs, k, dim=-1)
    topk_probs /= topk_probs.sum(dim=-1, keepdim=True)
    idx = torch.multinomial(topk_probs, num_samples=1)
    return topk_ids.gather(-1, idx).squeeze(-1)


def _sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    sorted_probs, sorted_ids = torch.sort(probs, dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)
    mask = cum_probs - sorted_probs > p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_ids.gather(-1, idx).squeeze(-1)


class Generator:
    def __init__(self, model, model_dtype: torch.dtype, bos_id: int, eos_id: int, pad_id: int) -> None:
        self.model = model
        self.model_dtype = model_dtype
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _seq_lens(self, tokens: torch.Tensor, attn_bias_type: str) -> list[int]:
        if attn_bias_type == "doc_causal":
            return tokens_to_seq_lens(tokens, self.eos_id)
        return [tokens.shape[1]] * tokens.shape[0]

    @torch.inference_mode()
    def get_logprobs(
        self, prompt_tokens: list[list[int]], max_prompt_len: int = 256,
        attn_bias_type: str = "causal",
    ) -> tuple[list[list[int]], list[list[float]], list[list[bool]]]:
        """Teacher-forced per-token log-probs for a batch of sequences.

        Returns (tokens, token_logprobs, greedy) where, for each sequence,
        token_logprobs[j] is logp(tokens[j+1] | tokens[:j+1]) and greedy[j] is
        True iff tokens[j+1] is the argmax of that distribution. Both are length
        orig_seq_len (= padded_batch_len - 1); callers slice their own range.
        One full forward pass, computed exactly as in training.
        """
        self.model.eval()
        tokens = [torch.tensor(t[-max_prompt_len:], device=self.device) for t in prompt_tokens]
        padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
        input_ids = padded[:, :-1]
        mask = self.model.create_attention_mask(input_ids, seq_lens=self._seq_lens(input_ids, attn_bias_type),
                                                attn_bias_type=attn_bias_type, compile_mask=False)
        logprobs = self.model(input_ids, mask).log_softmax(dim=-1)            # (bsz, L, V)
        targets = padded[:, 1:]
        greedy = logprobs.argmax(dim=-1) == targets
        token_logprobs = logprobs.gather(2, targets[..., None]).squeeze(-1)
        return [t[-max_prompt_len:] for t in prompt_tokens], token_logprobs.tolist(), greedy.tolist()

    def _steps(self, prompt_tokens, max_prompt_len, max_gen_len, attn_bias_type, choose):
        """Run generation, yielding (position, next tokens, which rows generated them).

        Rows whose prompt is longer than the shortest one are teacher-forced through
        their remaining prompt tokens, one decoding step at a time."""
        self.model.eval()
        prompt_tokens = [t[-max_prompt_len:] for t in prompt_tokens]
        bsz = len(prompt_tokens)
        prompt_lens = torch.tensor([len(t) for t in prompt_tokens], device=self.device)
        start, end = int(prompt_lens.min()), int(prompt_lens.max()) + max_gen_len
        tokens = torch.full((bsz, end), self.pad_id, device=self.device, dtype=torch.long)
        for k, toks in enumerate(prompt_tokens):
            tokens[k, :len(toks)] = torch.tensor(toks, dtype=torch.long)
        # By length, not by comparing with pad_id: pad may equal EOS, which a
        # multi-document prompt contains.
        in_prompt = torch.arange(end, device=self.device).view(1, -1) < prompt_lens.view(-1, 1)
        cache = self.model.create_kv_cache(bsz, end, self.model_dtype, device=self.device)

        prev = 0
        for cur in range(start, end):
            chunk = tokens[:, prev:cur]
            mask = self.model.create_attention_mask(chunk, seq_lens=self._seq_lens(chunk, attn_bias_type),
                                                    start_pos=prev, attn_bias_type=attn_bias_type,
                                                    compile_mask=False)
            logits, cache = self.model(chunk, cache=cache, mask=mask)
            # A row whose last token was EOS starts a new document with its next token.
            self.model.end_documents(cache, chunk[:, -1] == self.eos_id)
            next_token = torch.where(in_prompt[:, cur], tokens[:, cur], choose(logits[:, -1]))
            tokens[:, cur] = next_token
            prev = cur
            yield cur, next_token, ~in_prompt[:, cur], tokens

    def _chooser(self, use_sampling, temperature, top_k, top_p):
        def choose(logits):
            if not use_sampling:
                return torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits / temperature, dim=-1)
            if top_p > 0.0:
                return _sample_top_p(probs, top_p)
            if top_k > 0:
                return _sample_top_k(probs, top_k)
            return torch.multinomial(probs, num_samples=1).reshape(-1)
        return choose

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: list[list[int]],
        max_prompt_len: int = 256,
        max_gen_len: int = 256,
        *,
        use_sampling: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        remove_prompts: bool = True,
        stop_at_eos: bool = True,
        attn_bias_type: str = "causal",
    ) -> list[list[int]]:
        """Continuations of `prompt_tokens`, up to max_gen_len tokens each. With
        stop_at_eos, a row ends at its first generated EOS (excluded) and generation
        stops once every row has one; otherwise it runs on across documents."""
        bsz = len(prompt_tokens)
        done = torch.zeros(bsz, dtype=torch.bool, device=self.device)
        tokens = None
        for _, next_token, generated, tokens in self._steps(
                prompt_tokens, max_prompt_len, max_gen_len, attn_bias_type,
                self._chooser(use_sampling, temperature, top_k, top_p)):
            done |= generated & (next_token == self.eos_id)
            if stop_at_eos and done.all():
                break

        results = []
        for t, pt in zip(tokens, prompt_tokens):
            pt = pt[-max_prompt_len:]
            out = t[len(pt) if remove_prompts else 0:len(pt) + max_gen_len].tolist()
            if stop_at_eos:
                generated_part = out[len(pt):] if not remove_prompts else out
                if self.eos_id in generated_part:
                    cut = generated_part.index(self.eos_id) + (0 if remove_prompts else len(pt))
                    out = out[:cut]
            results.append(out)
        return results

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt_tokens: list[list[int]],
        max_prompt_len: int = 256,
        max_gen_len: int = 256,
        *,
        use_sampling: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        attn_bias_type: str = "causal",
    ) -> Iterator[torch.Tensor]:
        """Yield the next token of every row, one step at a time."""
        for _, next_token, _, _ in self._steps(prompt_tokens, max_prompt_len, max_gen_len, attn_bias_type,
                                               self._chooser(use_sampling, temperature, top_k, top_p)):
            yield next_token
