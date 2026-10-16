# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""SFT data pipeline.

Reads JSONL files where each line is a chat conversation:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Tokenizes with tokenizer.chat_template ("User: ... Assistant: ..."), setting mask=True
only on assistant answers and the final EOS. Packs conversations into fixed-length
sequences without wrapping (wrap=False): if a conversation doesn't fit the remaining
space, the rest of the sequence is padded and the conversation starts fresh in the next
one. Multiple conversations within one sequence are isolated by the doc-causal attention
mask (each conversation ends with eos_id, which tokens_to_seq_lens uses as boundary).
"""

import logging

from tokenizer.chat_template import apply_chat_template
from dataloader.streaming import JsonlSource, Stage

logger = logging.getLogger()


def _make_tokenize_fn(tokenizer):
    def fn(item: dict) -> dict | None:
        messages = item.get("messages")
        if not messages:
            return None
        try:
            tokens, mask = apply_chat_template(messages, tokenizer)
        except (ValueError, KeyError):
            return None
        if not tokens or not any(mask):
            return None
        return {"tokens": tokens, "mask": mask}
    return fn


def build_sft_pipeline(
    path: str,
    tokenizer,
    seq_len: int,
    batch_size: int,
    dp_rank: int,
    dp_world_size: int,
    seed: int,
    shuffle_buffer_size: int = 512,
) -> Stage:
    """JSONL directory → batched SFT sequences with per-token loss mask."""
    return (
        JsonlSource(path, dp_rank, dp_world_size)
        .shuffle(buffer_size=shuffle_buffer_size, seed=seed)
        .map(_make_tokenize_fn(tokenizer))
        .pack(seq_len + 1, wrap=False, pad=tokenizer.pad_id)
        .batch(batch_size)
    )
