# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Multi-head softmax attention over one memory scope.

Independent q/k/v projections, 1/sqrt(d) scale, plain values (a query may read its own
position), RoPE or no position encoding. The scope (model.common.mask.memory_mask) makes
it a sliding-window attention ("short"), a causal attention ("long", skip 0) or a
reach-past-the-recent attention ("long", skip > 0), or an attention over earlier
documents ("cluster"). The Transformer family uses the first two; the Mosaics family
may use any of them in place of a Mosaics memory.

RoPE positions come from Positions.tokenid: within the document under "doc_causal",
within the row otherwise, in training and generation alike.
"""

import torch
import torch.nn.functional as F
from torch import nn

from model.common.flex import flex_attention
from model.common.init import init_linear
from model.common.mask import FlexAttentionBias, reshape_for_flex_attention
from model.common.positions import KVCache, Positions
from model.common.rope import apply_rotary_emb, precompute_freqs_cis, shared_freqs_cis

_FLEX_OPTIONS = {"FORCE_USE_FLEX_ATTENTION": True}


class Attention(nn.Module):
    see_self = True           # plain values
    uses_mask = True

    def __init__(self, scope: str, dim: int, head_dim: int, n_heads: int, *,
                 window: int = 0, skip: int = 0, cluster_mem_skip: bool = True,
                 rope_theta: float | None = None, memory_gate: bool = False,
                 max_seq_len: int = 4096) -> None:
        """rope_theta None means no position encoding. `window` (short) and `skip`
        (long) are the inference-time scope; a training step's mask may use a
        randomized skip."""
        super().__init__()
        assert scope in ("short", "long", "cluster"), scope
        self.scope = scope
        self.dim, self.head_dim, self.n_heads = dim, head_dim, n_heads
        self.window, self.skip, self.cluster_mem_skip = window, skip, cluster_mem_skip
        self.memory_gate = memory_gate
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len

        H, d = n_heads, head_dim
        self.wq = nn.Linear(dim, H * d, bias=False)
        self.wk = nn.Linear(dim, H * d, bias=False)
        # [values | per-head read-gate logits]
        self.wv = nn.Linear(dim, H * d + (H if memory_gate else 0), bias=False)
        self.wo = nn.Linear(H * d, dim, bias=False)
        if rope_theta is not None:
            self.register_buffer("freqs_cis", precompute_freqs_cis(d, max_seq_len, rope_theta),
                                 persistent=False)

    def init_cache(self, bsz: int, max_seq_len: int, dtype, device) -> KVCache:
        shape = (bsz, max_seq_len, self.n_heads, self.head_dim)
        return (torch.empty(shape, dtype=dtype, device=device),
                torch.empty(shape, dtype=dtype, device=device), 0)

    def ensure_freqs_cis(self, needed: int, device: torch.device) -> None:
        """Extend the RoPE table to `needed` positions, e.g. to evaluate beyond the
        training length. Must run outside compiled code (create_attention_mask calls
        it): an index past the table end would otherwise be silently clamped."""
        if self.rope_theta is not None and self.freqs_cis.shape[0] < needed:
            self.freqs_cis = shared_freqs_cis(self.head_dim, needed, self.rope_theta, device)
            self.max_seq_len = needed

    def _decode_keys(self, k, v, pos: Positions):
        """The cache entries a decoding query (the last one) may read, and a per-row key
        mask when rows differ; None if it reads nothing."""
        kv_len = k.shape[1]
        if self.scope == "cluster":
            if pos.doc_start is None:                # no document structure
                return None
            if not self.cluster_mem_skip:            # the whole row, itself included
                return k, v, None
            ok = torch.arange(kv_len, device=k.device).view(1, -1) < pos.doc_start.long()
            return k, v, ok                          # earlier documents only
        if self.scope == "short":
            lo, hi = (max(kv_len - self.window, 0) if self.window > 0 else 0), kv_len
        else:                                        # long: at least `skip` back
            lo, hi = 0, kv_len - self.skip
        if hi <= lo:
            return None
        key_ok = None
        if pos.doc_start is not None:                # only the current document
            key_ok = torch.arange(lo, hi, device=k.device).view(1, -1) >= pos.doc_start.long()
        return k[:, lo:hi], v[:, lo:hi], key_ok

    def forward(self, x: torch.Tensor, mask: FlexAttentionBias | None, pos: Positions,
                cache: KVCache | None = None) -> tuple[torch.Tensor, KVCache | None]:
        bsz, seq_len, _ = x.shape
        H, d = self.n_heads, self.head_dim
        decoding = mask is None and cache is not None
        if mask is None and not decoding:
            return torch.zeros_like(x), cache        # nothing in scope under this bias type

        q = self.wq(x).view(bsz, seq_len, H, d)
        k = self.wk(x).view(bsz, seq_len, H, d)
        v = self.wv(x)
        gate = None
        if self.memory_gate:
            v, gate_logits = torch.split(v, [H * d, H], dim=2)
            gate = torch.sigmoid(gate_logits)
        v = v.view(bsz, seq_len, H, d)
        if self.rope_theta is not None:
            index = (pos.tokenid.reshape(-1).long() - 1).clamp(min=0)    # 1-based -> 0-based
            q, k = apply_rotary_emb(q, k, self.freqs_cis[index].view(bsz, seq_len, -1))

        if cache is not None:
            cache_k, cache_v, n = cache
            assert n == 0 or decoding, "prefill into a non-empty cache is not supported"
            cache_k[:, n:n + seq_len] = k.detach()
            cache_v[:, n:n + seq_len] = v.detach()
            cache = (cache_k, cache_v, n + seq_len)

        if not decoding:
            q, k, v = reshape_for_flex_attention(mask, q, k, v)
            out = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                 block_mask=mask.block_mask, kernel_options=_FLEX_OPTIONS)
        else:
            readable = self._decode_keys(cache[0][:, :cache[2]].contiguous(),
                                         cache[1][:, :cache[2]].contiguous(), pos)
            if readable is None:
                return torch.zeros_like(x), cache
            k, v, key_ok = readable
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if key_ok is None:
                out = flex_attention(q, k, v, block_mask=None, kernel_options=_FLEX_OPTIONS)
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_ok.view(bsz, 1, 1, -1))
                # A row that may read nothing (SDPA gives NaN there) contributes zero.
                out = torch.where(key_ok.any(dim=1).view(bsz, 1, 1, 1), out, torch.zeros_like(out))

        out = out.transpose(1, 2).reshape(bsz, seq_len, H, d)
        if gate is not None:
            out = out * gate.view(bsz, seq_len, H, 1)
        return self.wo(out.reshape(bsz, seq_len, H * d)), cache

    def init_weights(self, init_std: float | None = None, factor: float = 1.0) -> None:
        init_linear([self.wq, self.wk, self.wv, self.wo], init_std, self.dim, factor)
        # The RoPE table is a non-persistent buffer: rebuild it after meta-device
        # construction, and after .to(dtype), which casts complex tensors to real.
        if self.rope_theta is not None:
            with torch.device(self.freqs_cis.device):
                self.freqs_cis = precompute_freqs_cis(self.head_dim, self.max_seq_len, self.rope_theta)
