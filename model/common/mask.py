# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Attention masks: what each memory may read in a packed sequence of documents.

`memory_mask` builds the FlexAttention block mask for one memory scope; the `_*_mod`
functions are the elementwise mask rules it composes.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    create_block_mask,
)


@dataclass
class FlexAttentionBias:
    block_mask: BlockMask
    bias_type: str
    window_size: int | None = None        # short: the window; long: the skip (incl. random)


def tokens_to_seq_lens(tokens: torch.Tensor, eos_id: int) -> list[int]:
    """Document lengths of a (bsz, seq_len) batch, row-major: a document ends at an EOS
    (which belongs to it) or at the end of its row."""
    assert tokens.ndim == 2
    mask = tokens == eos_id
    mask[:, -1] = True
    row, col = torch.where(mask)
    seqlens = (col[1:] - col[:-1]) + (row[1:] - row[:-1]) * mask.shape[1]
    return [int(col[0].item() + 1), *seqlens.tolist()]


def _lens_to_start_ids(lens: torch.Tensor) -> torch.Tensor:
    doc_start = lens.cumsum(0)
    doc_start = doc_start.roll(1)
    doc_start[0] = 0
    return doc_start


def _lens_to_local_ids(lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert lens.ndim == 1
    total_seqlen = lens.sum()
    doc_id = torch.repeat_interleave(lens)
    doc_start = _lens_to_start_ids(lens)
    tok_doc_start = doc_start[doc_id]
    tok_id = torch.arange(total_seqlen, device=lens.device) - tok_doc_start
    return doc_id, tok_id


def _generate_doc_mask_mod(
    mask_mod: _mask_mod_signature,
    lens: torch.Tensor,
    kv_lens: torch.Tensor | None = None,
) -> _mask_mod_signature:
    kv_lens = kv_lens if kv_lens is not None else lens
    q_document_id, q_token_id = _lens_to_local_ids(lens)
    kv_document_id, kv_token_id = _lens_to_local_ids(kv_lens)
    q_max_idx = lens.sum() - 1
    kv_max_idx = kv_lens.sum() - 1

    def doc_mask_mod(b, h, q_idx, kv_idx):
        q_idx_cap = torch.minimum(q_max_idx, q_idx)
        kv_idx_cap = torch.minimum(kv_max_idx, kv_idx)
        valid_idx = (q_idx <= q_max_idx) & (kv_idx <= kv_max_idx)
        same_doc = q_document_id[q_idx_cap] == kv_document_id[kv_idx_cap]
        q_logical = q_token_id[q_idx_cap]
        kv_logical = kv_token_id[kv_idx_cap]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask & valid_idx

    return doc_mask_mod


def _causal_shift_mod(shift: int) -> Callable:
    """Global memory mask: q_idx >= kv_idx + shift (skip recent window)."""
    def mask_mod(b: Any, h: Any, q_idx: Any, kv_idx: Any):
        return q_idx >= kv_idx + shift
    return mask_mod


def _sliding_window_offone_mod(window: int) -> Callable:
    """Local memory mask: off-one sliding window (diagonal masked out)."""
    def mask_mod(b: Any, h: Any, q_idx: Any, kv_idx: Any):
        return (q_idx > kv_idx) & (q_idx < kv_idx + window)
    return mask_mod


def _sliding_window_mod(window: int) -> Callable:
    """Local memory mask: sliding window (diagonal included)."""
    def mask_mod(b: Any, h: Any, q_idx: Any, kv_idx: Any):
        return (q_idx >= kv_idx) & (q_idx < kv_idx + window)
    return mask_mod


def _causal_offone_mod() -> Callable:
    """Full causal mask (diagonal excluded)."""
    def mask_mod(b: Any, h: Any, q_idx: Any, kv_idx: Any):
        return q_idx > kv_idx
    return mask_mod


def _causal_mod() -> Callable:
    """Full causal mask (diagonal included)."""
    def mask_mod(b: Any, h: Any, q_idx: Any, kv_idx: Any):
        return q_idx >= kv_idx
    return mask_mod


def _cross_doc_causal_mod(lens: torch.Tensor, seq_len: int) -> Callable:
    """Cluster memory mask: attend to all tokens in strictly earlier documents
    within the same batch element. seq_len is the per-batch-element length
    so that we don't cross batch boundaries in the flattened sequence."""
    q_document_id, _ = _lens_to_local_ids(lens)
    kv_document_id, _ = _lens_to_local_ids(lens)
    q_max_idx = lens.sum() - 1

    def mask_mod(b: Any, h: Any, q_idx: Any, kv_idx: Any):
        q_idx_cap = torch.minimum(q_max_idx, q_idx)
        kv_idx_cap = torch.minimum(q_max_idx, kv_idx)
        valid_idx = (q_idx <= q_max_idx) & (kv_idx <= q_max_idx)
        same_batch = (q_idx_cap // seq_len) == (kv_idx_cap // seq_len)
        cross_doc = q_document_id[q_idx_cap] > kv_document_id[kv_idx_cap]
        return same_batch & cross_doc & valid_idx

    return mask_mod


def _full_causal_mod(seq_len: int) -> Callable:
    """Cluster mask with cluster_mem_skip=False for an "attention" cluster: every earlier
    token in the same batch row, current AND previous documents, diagonal INCLUDED. Safe
    only because an attention block's values are plain projections; the Mosaics block's
    look-ahead values need _full_causal_offone_mod instead."""
    def mask_mod(b: Any, h: Any, q_idx: Any, kv_idx: Any):
        same_batch = (q_idx // seq_len) == (kv_idx // seq_len)
        return same_batch & (q_idx >= kv_idx)

    return mask_mod


def _full_causal_offone_mod(seq_len: int) -> Callable:
    """Cluster mask with cluster_mem_skip=False for a Mosaics cluster: every earlier
    token in the same row, current and previous documents, diagonal EXCLUDED (the
    look-ahead value v_t holds x_{t+1}). seq_len is the row length, so attention stays
    within a batch element."""
    def mask_mod(b: Any, h: Any, q_idx: Any, kv_idx: Any):
        same_batch = (q_idx // seq_len) == (kv_idx // seq_len)
        return same_batch & (q_idx > kv_idx)

    return mask_mod


def memory_mask(
    scope: str,
    see_self: bool,
    tokens: torch.Tensor,
    q_lens: torch.Tensor,
    attn_bias_type: str,
    *,
    kv_lens: torch.Tensor | None = None,
    window: int = 0,
    skip: int = 0,
    cluster_mem_skip: bool = True,
) -> FlexAttentionBias | None:
    """The block mask of one memory: which earlier positions each query may read.

    scope:
      "short"   the last `window` tokens (0 = all earlier tokens)
      "long"    tokens at least `skip` back (skip 0 with see_self: plain causal attention)
      "cluster" strictly earlier documents (cluster_mem_skip), else the whole row;
                only defined under "doc_causal" (returns None otherwise)
    see_self: whether a query may read its own position. False for Mosaics memories,
      whose value v_t holds x_{t+1}, i.e. the token being predicted.
    attn_bias_type: "doc_causal" confines short and long to the current document;
      "causal" lets them span the packed row.
    """
    bsz, seq_len = tokens.shape
    device = tokens.device
    if scope == "cluster":
        if attn_bias_type != "doc_causal":
            return None
        total = int(q_lens.sum().item())
        if cluster_mem_skip:
            mod = _cross_doc_causal_mod(q_lens, seq_len)
        else:
            mod = (_full_causal_mod if see_self else _full_causal_offone_mod)(seq_len)
        return FlexAttentionBias(_create_block_mask(mod, None, None, total, total, device),
                                 bias_type=attn_bias_type)

    if scope == "short":
        size = window
        if window > 0:
            inner = (_sliding_window_mod if see_self else _sliding_window_offone_mod)(window)
        else:
            inner = _causal_mod() if see_self else _causal_offone_mod()
    elif scope == "long":
        size = skip
        inner = _causal_shift_mod(skip)
    else:
        raise ValueError(f"unknown memory scope {scope!r}")

    if attn_bias_type == "doc_causal":
        kv_lens = kv_lens if kv_lens is not None else q_lens
        mod = _generate_doc_mask_mod(inner, q_lens, kv_lens)
        block_mask = _create_block_mask(mod, None, None, int(q_lens.sum().item()),
                                        int(kv_lens.sum().item()), device)
    elif attn_bias_type == "causal":
        block_mask = _create_block_mask(inner, None, None, seq_len, seq_len, device)
    else:
        raise ValueError(f"Unsupported attn_bias_type: {attn_bias_type}")
    return FlexAttentionBias(block_mask, bias_type=attn_bias_type, window_size=size)


def reshape_for_flex_attention(
    attn_bias: FlexAttentionBias,
    *tensors: torch.Tensor,
) -> list[torch.Tensor]:
    to_transform = list(tensors)
    if attn_bias.bias_type.startswith("doc_causal"):
        to_transform = [t.view(1, -1, *t.shape[2:]) for t in to_transform]
    return to_transform


_BLOCK_SIZE = 128
_DENSE_MASK_LIMIT = 128 * 1024  # 128k tokens — above this, dense mask would OOM

# Whether to torch.compile create_block_mask. Compiling is worth it for the
# fixed-shape training loop (compiled once, reused every step), but is a
# disaster for generation/eval where prompt lengths vary: each new shape
# triggers a fresh (slow) ptxas compile. Generation/eval set this False.
_COMPILE_BLOCK_MASK = True


def set_block_mask_compile(flag: bool) -> None:
    global _COMPILE_BLOCK_MASK
    _COMPILE_BLOCK_MASK = flag


def _create_block_mask(mask_mod, B, H, total_q, total_kv, device):
    """Create BlockMask efficiently. For long sequences, avoids the dense
    (total_q, total_kv) bool tensor by tiling the mask_mod evaluation over query
    blocks and reducing to block granularity -- exact for any mask_mod -- then
    constructing BlockMask directly."""
    if max(total_q, total_kv) <= _DENSE_MASK_LIMIT:
        compile = _COMPILE_BLOCK_MASK and total_q >= _BLOCK_SIZE * 2 and total_kv >= _BLOCK_SIZE * 2
        return create_block_mask(mask_mod, B, H, total_q, total_kv, device, _compile=compile)#, compute_q_blocks=True)

    Q_BLOCKS = (total_q + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    KV_BLOCKS = (total_kv + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    # Exact block activity: a (q_block, kv_block) is active iff ANY (q, kv) inside
    # it satisfies mask_mod. Evaluate mask_mod on a [rows, total_kv] grid per query
    # tile and reduce with .any() over the 128x128 tiles. Exact for any mask_mod --
    # unlike corner sampling, which drops document-boundary-straddling blocks for
    # the same_doc global/local masks (the interior maximizer is never a corner).
    # Tile over query blocks so peak memory is O(rows * total_kv), not the dense
    # O(total_q * total_kv). mask_mod is only ever evaluated on in-range indices.
    KV_PAD = KV_BLOCKS * _BLOCK_SIZE
    # Cap the query tile so the [rows, total_kv] bool grid stays ~128 MB.
    q_tile_blocks = max(1, min(Q_BLOCKS, (128 * 1024 * 1024) // (_BLOCK_SIZE * max(total_kv, 1))))

    block_active = torch.zeros(Q_BLOCKS, KV_BLOCKS, dtype=torch.bool, device=device)
    kv_pos = torch.arange(total_kv, device=device).unsqueeze(0)              # [1, total_kv]
    for qb0 in range(0, Q_BLOCKS, q_tile_blocks):
        q0 = qb0 * _BLOCK_SIZE
        q1 = min(q0 + q_tile_blocks * _BLOCK_SIZE, total_q)
        rows = q1 - q0
        qtb = (rows + _BLOCK_SIZE - 1) // _BLOCK_SIZE                        # q-blocks in this tile
        q_pos = torch.arange(q0, q1, device=device).unsqueeze(1)            # [rows, 1]
        m = mask_mod(0, 0, q_pos, kv_pos).to(torch.bool)                    # [rows, total_kv]

        # any() over kv within each 128-block (pad to a block multiple with False).
        if total_kv == KV_PAD:
            kv_red = m.reshape(rows, KV_BLOCKS, _BLOCK_SIZE).any(dim=2)      # [rows, KV_BLOCKS]
        else:
            buf = torch.zeros(rows, KV_PAD, dtype=torch.bool, device=device)
            buf[:, :total_kv] = m
            kv_red = buf.reshape(rows, KV_BLOCKS, _BLOCK_SIZE).any(dim=2)

        # any() over q within each 128-block (pad the row dim with False).
        q_buf = torch.zeros(qtb * _BLOCK_SIZE, KV_BLOCKS, dtype=torch.bool, device=device)
        q_buf[:rows] = kv_red
        block_active[qb0:qb0 + qtb] = q_buf.reshape(qtb, _BLOCK_SIZE, KV_BLOCKS).any(dim=1)

    kv_num_blocks = block_active.sum(-1).int()
    max_kv = int(kv_num_blocks.max().item())
    if max_kv == 0:
        max_kv = 1
    kv_indices = torch.zeros(Q_BLOCKS, max_kv, dtype=torch.int32, device=device)

    for q in range(Q_BLOCKS):
        active = block_active[q].nonzero(as_tuple=True)[0].int()
        if len(active) > 0:
            kv_indices[q, :len(active)] = active

    # q_indices: transpose of kv_indices — for each KV block, which Q blocks attend to it.
    # Required by the backward pass.
    q_num_blocks = block_active.sum(0).int()  # [KV_BLOCKS]
    max_q = int(q_num_blocks.max().item())
    if max_q == 0:
        max_q = 1
    q_indices = torch.zeros(KV_BLOCKS, max_q, dtype=torch.int32, device=device)

    for kv in range(KV_BLOCKS):
        active = block_active[:, kv].nonzero(as_tuple=True)[0].int()
        if len(active) > 0:
            q_indices[kv, :len(active)] = active

    kv_num_blocks = kv_num_blocks.view(1, 1, Q_BLOCKS)
    kv_indices = kv_indices.view(1, 1, Q_BLOCKS, max_kv)
    q_num_blocks = q_num_blocks.view(1, 1, KV_BLOCKS)
    q_indices = q_indices.view(1, 1, KV_BLOCKS, max_q)

    return BlockMask(
        seq_lengths=(total_q, total_kv),
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=None,
        full_kv_indices=None,
        q_num_blocks=q_num_blocks,
        q_indices=q_indices,
        full_q_num_blocks=None,
        full_q_indices=None,
        BLOCK_SIZE=(_BLOCK_SIZE, _BLOCK_SIZE),
        mask_mod=mask_mod,
    )
