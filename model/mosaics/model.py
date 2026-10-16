# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""MemoryMosaics: the language model of the Memory Mosaics family.

Each block (model.mosaics.block) holds up to three in-context memories and a persistent
memory:

    short     the last few tokens            Mosaics memory, attention, or DeltaNet
    long      the rest of the document       Mosaics memory or attention
    cluster   earlier, related documents     Mosaics memory or attention
    persistent knowledge of the training data, the FFN

Memory Mosaics, MMv2, MM+, two-stage MM+, Transformer+ and two-stage Transformer+ are
configurations of it (configs/mosaics/).
"""

import random

import torch

from model.common.lm import LanguageModel
from model.common.mask import memory_mask
from model.mosaics.block import MosaicBlock
from model.mosaics.config import MosaicsConfig


class MemoryMosaics(LanguageModel):
    def __init__(self, args: MosaicsConfig) -> None:
        super().__init__(args, [MosaicBlock(args, layer_id) for layer_id in range(args.n_layers)])

    def build_masks(self, tokens, q_lens, kv_lens, attn_bias_type, step, seed) -> dict:
        args = self.args
        # Training augmentation: a random extra long-memory skip, drawn per (seed, step, rank).
        long_skip = args.long_mem_skip_size
        if self.training and args.random_long_mem_skip_size > 0:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            rng = random.Random(seed * 1_000_000_000 + step * 1000 + rank)
            long_skip += rng.randint(0, args.random_long_mem_skip_size)

        block = self.blocks()[0]        # every block has the same memories
        masks = {}
        sizes = {"short": dict(window=args.short_mem_window_size), "long": dict(skip=long_skip),
                 "cluster": dict(cluster_mem_skip=args.cluster_mem_skip)}
        for scope, size in sizes.items():
            memory = getattr(block, f"{scope}_memory")
            if memory is None or not memory.uses_mask or (scope == "cluster" and args.disable_cluster):
                continue
            mask = memory_mask(scope, memory.see_self, tokens, q_lens, attn_bias_type, kv_lens=kv_lens, **size)
            if mask is not None:
                masks[scope] = mask
        return masks
