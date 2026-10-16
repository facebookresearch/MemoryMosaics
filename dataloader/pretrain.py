# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Pretrain data pipeline, per source: JSONL → text → shuffle → tokenize → pack; then
weighted sampling across sources → batch."""

import logging

from dataloader.streaming import JsonlSource, Sample, Stage

logger = logging.getLogger()


def _extract_text(item: dict) -> dict | None:
    text = item.get("text", item.get("content", ""))
    if not text:
        return None
    result = {"text": text}
    if "eidx" in item:
        result["eidx"] = item["eidx"]
    return result


def _make_tokenize_fn(tokenizer, add_bos: bool, add_eos: bool):
    def fn(item: dict) -> dict | None:
        text = item["text"] if isinstance(item, dict) else item
        tokens = tokenizer.encode(text)
        if not tokens:
            return None
        tokens = [tokenizer.bos_id] * add_bos + tokens + [tokenizer.eos_id] * add_eos
        result = {"tokens": tokens, "mask": [True] * len(tokens)}
        if isinstance(item, dict) and "eidx" in item:
            result["eidx"] = item["eidx"]
        return result
    return fn


def build_pretrain_source(
    path: str,
    tokenizer,
    seq_len: int,
    dp_rank: int,
    dp_world_size: int,
    seed: int,
    shuffle_buffer_size: int = 64,
    add_bos: bool = True,
    add_eos: bool = True,
) -> Stage:
    """One source directory → endless packed sequences of seq_len + 1 tokens."""
    return (
        JsonlSource(path, dp_rank, dp_world_size)
        .map(_extract_text)
        .shuffle(buffer_size=shuffle_buffer_size, seed=seed)
        .map(_make_tokenize_fn(tokenizer, add_bos, add_eos))
        .pack(seq_len + 1, wrap=True)
    )


def build_pretrain_pipeline(
    sources: list[tuple[str, float]],
    tokenizer,
    seq_len: int,
    batch_size: int,
    dp_rank: int,
    dp_world_size: int,
    seed: int,
    shuffle_buffer_size: int = 64,
    add_bos: bool = True,
    add_eos: bool = True,
) -> Stage:
    """Multiple source directories with weights → batched training data."""
    paths = [p for p, _ in sources]
    weights = [w for _, w in sources]

    datasets = [
        build_pretrain_source(
            path, tokenizer, seq_len,
            dp_rank, dp_world_size,
            seed=seed + i * 1000,
            shuffle_buffer_size=shuffle_buffer_size,
            add_bos=add_bos, add_eos=add_eos,
        )
        for i, path in enumerate(paths)
    ]

    return Sample(datasets, weights, seed=seed).batch(batch_size)
