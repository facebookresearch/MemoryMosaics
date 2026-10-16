# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Rotary position embedding (RoPE), shared by every attention block in the repo."""

import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Complex rotations e^{i * t * theta^(-2j/dim)} for positions t < end, (end, dim // 2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate (..., seq, heads, head_dim) queries and keys by (..., seq, head_dim // 2)."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(-2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# Read-only tables, identical across layers, so layers share one per (head_dim, theta,
# device) instead of each holding a copy: at 786k positions a table is 0.38 GiB, and 24
# private copies per memory would not fit next to the model.
_SHARED_TABLES: dict = {}


def shared_freqs_cis(head_dim: int, needed: int, theta: float, device) -> torch.Tensor:
    """A table covering at least `needed` positions, shared by every caller with the same key."""
    key = (head_dim, float(theta), str(device))
    table = _SHARED_TABLES.get(key)
    if table is None or table.shape[0] < needed:
        size = 1 << max(needed - 1, 1).bit_length()   # grow geometrically (generation adds 1 per step)
        table = precompute_freqs_cis(head_dim, size, theta).to(device=device)
        _SHARED_TABLES[key] = table
    return table
