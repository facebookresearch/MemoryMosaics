# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""DeltaNet: a recurrent language model. Every layer is a MixerBlock around a gated
DeltaNet (model.common.deltanet), so its memory is a fixed-size state per head."""

from model.common.block import MixerBlock
from model.common.deltanet import DeltaNet
from model.common.lm import LanguageModel
from model.rnn.config import RNNConfig


class RNN(LanguageModel):
    def __init__(self, args: RNNConfig) -> None:
        super().__init__(args, [
            MixerBlock(DeltaNet(args.dim, args.head_dim, args.heads, memory_gate=args.memory_gate), args)
            for _ in range(args.n_layers)])

    def build_masks(self, tokens, q_lens, kv_lens, attn_bias_type, step, seed) -> dict:
        return {}     # the recurrence takes its document scope from the positions
