# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""The block of the Transformer and RNN families: one sequence mixer and an FFN,

    x += mixer(norm(x));  x += ffn(norm(x))

where the mixer is an Attention (full or sliding-window) or a DeltaNet.
"""

import torch
from torch import nn

from model.common.config import BaseConfig
from model.common.feed_forward import FeedForward, chunked_ffn_residual
from model.common.norm import RMSNorm
from model.common.positions import Positions


def feed_forward(args: BaseConfig) -> FeedForward:
    return FeedForward(dim=args.dim, hidden_dim=args.ffn_exp * args.dim,
                       multiple_of=args.multiple_of, ffn_dim_multiplier=args.ffn_dim_multiplier)


class MixerBlock(nn.Module):
    def __init__(self, mixer: nn.Module, args: BaseConfig, mask_key: str | None = None) -> None:
        """`mask_key` names this block's mask in the model's masks (None: no mask)."""
        super().__init__()
        self.mask_key = mask_key
        self.mixer_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.mixer = mixer
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.feed_forward = feed_forward(args)

    def init_cache(self, bsz: int, max_seq_len: int, dtype, device):
        return self.mixer.init_cache(bsz, max_seq_len, dtype, device)

    def forward(self, x: torch.Tensor, masks: dict | None, pos: Positions, cache=None):
        mask = masks.get(self.mask_key) if (masks is not None and self.mask_key) else None
        out, cache = self.mixer(self.mixer_norm(x), mask, pos, cache)
        return chunked_ffn_residual(x + out, self.ffn_norm, self.feed_forward), cache

    def init_weights(self, init_std: float | None = None, factor: float = 1.0) -> None:
        self.mixer.init_weights(init_std, factor)
        self.mixer_norm.reset_parameters()
        self.feed_forward.init_weights(init_std, factor)
        self.ffn_norm.reset_parameters()
