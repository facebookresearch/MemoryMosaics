# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""The persistent memory: a SwiGLU feed-forward network."""

import os

import torch
from torch import nn
from torch.nn import functional as F

# Sequence-chunk the SwiGLU at inference. The FFN is position-wise, so slicing the
# sequence dimension is EXACT -- no cross-position term to lose. It matters at long
# eval lengths: w1(x), silu(...), w3(x) and their product are each
# seq_len * hidden_dim * 2 bytes, so at seq_len 786432 with hidden 5632 that is 8.25 GiB
# *four times over*, which OOMs an 80 GB card on its own.
# Only applied with grad disabled: under autograd every chunk's activations are retained
# for backward anyway, so chunking would cost Python overhead and save nothing.
FFN_CHUNK_TOKENS = int(os.environ.get("MOSAICS_FFN_CHUNK_TOKENS", 64 * 1024))


def chunked_ffn_residual(x: torch.Tensor, norm: nn.Module, ffn: nn.Module) -> torch.Tensor:
    """x + ffn(norm(x)), evaluated in sequence chunks at inference.

    Both RMSNorm and SwiGLU are position-wise (the norm reduces over the FEATURE dim
    only), so slicing the sequence is exact.

    They must be chunked TOGETHER. Chunking inside FeedForward alone does not help,
    because the caller evaluates norm(x) first and RMSNorm upcasts: x.float() and
    x.pow(2) are each seq_len*dim*4 bytes -- 6 GiB apiece at seq_len 786432, ~18 GiB
    transient -- before the FFN is even entered. That fp32 copy is what OOMs first.
    """
    if not (FFN_CHUNK_TOKENS > 0 and not torch.is_grad_enabled()
            and x.dim() == 3 and x.shape[1] > FFN_CHUNK_TOKENS):
        return x + ffn(norm(x))
    out = torch.empty_like(x)
    for i in range(0, x.shape[1], FFN_CHUNK_TOKENS):
        xi = x[:, i:i + FFN_CHUNK_TOKENS]
        out[:, i:i + FFN_CHUNK_TOKENS] = xi + ffn(norm(xi))
    return out


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float | None = None, factor: float = 1.0) -> None:
        in_init_std = (init_std or (self.dim ** (-0.5))) / factor
        for w in [self.w1, self.w3]:
            nn.init.trunc_normal_(
                w.weight, mean=0.0, std=in_init_std,
                a=-3 * in_init_std, b=3 * in_init_std,
            )
        out_init_std = (init_std or (self.hidden_dim ** (-0.5))) / factor
        nn.init.trunc_normal_(
            self.w2.weight, mean=0.0, std=out_init_std,
            a=-3 * out_init_std, b=3 * out_init_std,
        )
