# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""The Mosaics memory: Gaussian kernel regression over key/value pairs of the past.

For each head, a query q_T reads the memory as  sum_i softmax_i(q_T . k_i) v_i  where

    k_T = phi(x_T, x_{T-1}, ...)    a unit-norm leaky average of projected inputs
          (or phi(x_T) alone, with <scope>_key_feature: plain)
    q_T = beta(n) * k_T             the query is the key, scaled by a bandwidth
    beta(n) = b1 * n^a + b0         that sharpens as the memory size n grows
    v_T = psi(x_{T+1}, x_T)         unit-norm interpolation of the next and current
                                    projected input: the value looks one token ahead.

Because v_T carries x_{T+1}, a query never reads its own position.

One class serves all three scopes; the scope decides what the memory may read
(model.common.mask.memory_mask) and what n is:
    short    n = position in the document, capped at the window
    long     n = number of tokens behind the skip
    cluster  n = number of tokens in earlier documents
"""

import torch
import torch.nn.functional as F
from torch import nn

from model.common.flex import flex_attention
from model.common.init import init_linear
from model.common.mask import FlexAttentionBias, reshape_for_flex_attention
from model.common.positions import KVCache, Positions
from model.mosaics.config import MosaicsConfig
from model.mosaics.leaky_average import LeakyAverage

_EXP_CLIP = 15.0
_FLEX_OPTIONS = {"FORCE_USE_FLEX_ATTENTION": True}


def chunked_leaky_average(
    recurrence: LeakyAverage, xk: torch.Tensor, decay_logits: torch.Tensor,
    gate_logits: torch.Tensor, n_heads: int, head_dim: int, chunk_size: int, padding: int,
) -> torch.Tensor:
    """The leaky-average key over (bsz, seq_len, ...) inputs, computed in chunks.

    The sequence is cut into chunks of ~chunk_size tokens and each chunk restarts from
    zero after `padding` tokens of warm-up from the previous chunk (decay -inf on the pad
    resets the state), so the key at T only sums over the last chunk_size + padding inputs
    at most. padding=0 runs each chunk independently. A length that does not split evenly
    is right-padded, which leaves the outputs unchanged (the recurrence is causal).
    """
    bsz, seq_len, _ = xk.shape
    fac = max(seq_len // chunk_size, 1)
    step = -(-seq_len // fac)
    if step * fac != seq_len:
        right = step * fac - seq_len
        xk, decay_logits, gate_logits = (F.pad(t, (0, 0, 0, right)) for t in (xk, decay_logits, gate_logits))
    total = step * fac
    if padding > 0:
        def unfold(t, value=0.0):
            return (F.pad(t, pad=(0, 0, padding, 0), value=value)
                    .unfold(dimension=1, size=step + padding, step=step)
                    .transpose(2, 3).contiguous()
                    .view(bsz * fac, -1, t.shape[-1]))
        y = recurrence(unfold(xk), unfold(decay_logits, float("-inf")),
                       gate_logits=unfold(gate_logits))[:, padding:].contiguous()
    else:
        def chunk(t):
            return t.reshape(bsz * fac, step, t.shape[-1])
        y = recurrence(chunk(xk), chunk(decay_logits), gate_logits=chunk(gate_logits))
    return y.reshape(bsz, total, n_heads, head_dim)[:, :seq_len]


def lookahead_values(xv: torch.Tensor, interp: torch.Tensor) -> torch.Tensor:
    """v_t = normalize((1 - interp) * xv_{t+1} + interp * xv_t), per head. (bsz, T, H, d)."""
    H = xv.shape[2]
    w = interp.view(1, 1, H, 1)
    xv_next = F.pad(xv, pad=(0, 0, 0, 0, -1, 1))
    return F.normalize(xv_next * (1 - w) + xv * w, dim=-1)


class MosaicsMemory(nn.Module):
    see_self = False          # look-ahead values
    uses_mask = True

    def __init__(self, scope: str, cfg: MosaicsConfig) -> None:
        super().__init__()
        assert scope in ("short", "long", "cluster"), scope
        self.scope = scope
        dim, d, H = cfg.dim, cfg.head_dim, cfg.heads(scope)
        self.dim, self.head_dim, self.n_heads = dim, d, H
        self.window = cfg.short_mem_window_size
        self.skip = cfg.long_mem_skip_size
        self.cluster_mem_skip = cfg.cluster_mem_skip
        self.memory_gate = cfg.memory_gate
        self.recurrence_chunk_size = cfg.recurrence_chunk_size
        self.recurrence_padding = cfg.recurrence_padding
        self.leaky_key = cfg.leaky_key(scope)

        if self.leaky_key:
            # [key features | per-head decay logits | per-head input-gate logits]
            self.proj_kf = nn.Linear(dim, H * (d + 2), bias=False)
            self.recurrence = LeakyAverage(n_embd=H * d, n_head=H, n_decays_per_head=1, scale=1)
        else:
            self.proj_kf = nn.Linear(dim, H * d, bias=False)
        # [values | per-head read-gate logits]
        self.proj_v = nn.Linear(dim, H * d + (H if self.memory_gate else 0), bias=False)
        self.proj_out = nn.Linear(H * d, dim, bias=False)

        self.value_interp = nn.Parameter(torch.empty(H))          # psi's mix of x_t, x_{t+1}
        self.bandwidth_bias = nn.Parameter(torch.empty(H))        # log b0
        self.bandwidth_scale = nn.Parameter(torch.empty(H))       # log b1
        self.bandwidth_exponent = nn.Parameter(torch.empty(H))    # a (abs, capped at 1)
        self.value_scale = nn.Parameter(torch.empty(H))           # log output gain
        self._reset_scalars()

    def init_cache(self, bsz: int, max_seq_len: int, dtype, device) -> KVCache:
        shape = (bsz, max_seq_len, self.n_heads, self.head_dim)
        return (torch.empty(shape, dtype=dtype, device=device),
                torch.empty(shape, dtype=dtype, device=device), 0)

    # ── pieces of the computation ──

    def _bandwidth(self, n: torch.Tensor, bsz: int, seq_len: int) -> torch.Tensor:
        """beta(n) = b1 * n^a + b0 per head, shape (bsz, seq_len, H, 1).

        Computed in fp32 whatever the parameter dtype: beta is large (tens), the query is
        the key times beta, so bf16 rounding here would shift every attention score.
        """
        H = self.n_heads
        b0 = torch.exp(torch.clamp(self.bandwidth_bias.float(), max=_EXP_CLIP - 5)).view(1, 1, H, 1)
        b1 = torch.exp(torch.clamp(self.bandwidth_scale.float(), max=_EXP_CLIP - 5)).view(1, 1, H, 1)
        a = torch.clamp(self.bandwidth_exponent.float().abs(), max=1).view(1, 1, H, 1)
        n = n.reshape(bsz, seq_len, 1, 1).float()
        return b1 * torch.pow(n, a) + b0

    def _memory_size(self, pos: Positions, mask: FlexAttentionBias | None, generating: bool,
                     bsz: int, seq_len: int) -> torch.Tensor:
        """n, the number of entries this memory holds for each query."""
        if self.scope == "short":
            window = mask.window_size if mask is not None else self.window
            tok = pos.tokenid.reshape(bsz, seq_len)
            return tok if window <= 0 else torch.clamp(tok, max=window)
        if self.scope == "long":
            skip = mask.window_size if mask is not None else self.skip
            return torch.clamp(pos.tokenid.reshape(bsz, seq_len) - skip, min=0)
        if self.cluster_mem_skip or generating:
            return pos.cluster_tokenid
        # Whole-row cluster memory: n is the (0-based) position in the row.
        return torch.arange(seq_len, device=pos.cluster_tokenid.device,
                            dtype=pos.cluster_tokenid.dtype).repeat(bsz)

    def _keys(self, x: torch.Tensor, cache: KVCache | None) -> torch.Tensor:
        """Keys for x: chunked as in training without a cache; with one, the whole
        prompt (keeping the final recurrence state) or one step from that state."""
        bsz, seq_len, _ = x.shape
        H, d = self.n_heads, self.head_dim
        if not self.leaky_key:
            return F.normalize(self.proj_kf(x).view(bsz, seq_len, H, d), dim=-1)
        xk, decay, gate = torch.split(self.proj_kf(x), [H * d, H, H], dim=2)
        if cache is None:
            return chunked_leaky_average(self.recurrence, xk, decay, gate, H, d,
                                         self.recurrence_chunk_size, self.recurrence_padding)
        if cache[2] > 0:    # a decoding step; the state is updated in place
            return self.recurrence.forward_cache(
                xk.view(bsz, 1, H * d), decay, gate_logits=gate, y_prev=self._recurrence_state,
            ).view(bsz, 1, H, d)
        last = torch.full((bsz,), seq_len - 1, device=x.device, dtype=torch.long)
        xk, state = self.recurrence.forward_state_pad(xk, decay, gate_logits=gate, loc=last)
        self._recurrence_state = state.reshape(bsz, H * d)
        return xk.view(bsz, seq_len, H, d)

    def _decode_memory(self, k, v, pos: Positions):
        """The cache entries a decoding query may read -- not the last one, the query
        itself, whose value is written one step later -- and a per-row key mask when
        rows differ; None if it reads nothing."""
        kv_len = k.shape[2]
        if self.scope == "cluster":                  # each row's earlier documents
            ok = torch.arange(kv_len - 1, device=k.device).view(1, -1) < pos.doc_start.long()
            return k[:, :, :-1], v[:, :, :-1], ok
        if self.scope == "short":
            lo, hi = (max(kv_len - self.window, 0) if self.window > 0 else 0), kv_len - 1
        else:                                        # long: at least `skip` back
            lo, hi = 0, kv_len - self.skip
        if hi <= lo:
            return None
        key_ok = None
        if pos.doc_start is not None:                # only the current document
            key_ok = torch.arange(lo, hi, device=k.device).view(1, -1) >= pos.doc_start.long()
        return k[:, :, lo:hi], v[:, :, lo:hi], key_ok

    # ── forward ──

    def forward(self, x: torch.Tensor, mask: FlexAttentionBias | None, pos: Positions,
                cache: KVCache | None = None) -> tuple[torch.Tensor, KVCache | None]:
        bsz, seq_len, _ = x.shape
        H, d = self.n_heads, self.head_dim
        generating = cache is not None
        if self.scope == "cluster":
            if pos.cluster_tokenid is None or (mask is None and not generating):
                return torch.zeros_like(x), cache    # no earlier documents to read
            if generating and not self.cluster_mem_skip:
                raise NotImplementedError("generation with cluster_mem_skip: false")

        xv = self.proj_v(x)
        gate = None
        if self.memory_gate:
            xv, gate_logits = torch.split(xv, [H * d, H], dim=2)
            gate = torch.sigmoid(gate_logits)
        xv = xv.view(bsz, seq_len, H, d)

        xk = self._keys(x, cache)
        decoding = generating and cache[2] > 0
        if decoding:        # completes the previous token's value, now that x_{T+1} is known
            w = self.value_interp.view(1, 1, H, 1)
            v = F.normalize(xv * (1 - w) + self._prev_v * w, dim=-1)
        else:
            v = lookahead_values(xv, self.value_interp)
        if generating:
            self._prev_v = xv[:, -1:].clone()

        q = xk * self._bandwidth(self._memory_size(pos, mask, generating, bsz, seq_len),
                                 bsz, seq_len).to(xk.dtype)

        if generating:
            cache_k, cache_v, n = cache
            # The value of the previous token is completed only now (decoding).
            off = 1 if decoding else 0
            cache_k[:, n:n + seq_len] = xk.detach()
            cache_v[:, n - off:n - off + seq_len] = v.detach()
            cache = (cache_k, cache_v, n + seq_len)
            if mask is None:
                xk, v = cache_k[:, :n + seq_len].contiguous(), cache_v[:, :n + seq_len].contiguous()

        if mask is not None:
            q, k, v = reshape_for_flex_attention(mask, q, xk, v)
            out = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                 block_mask=mask.block_mask, scale=1, kernel_options=_FLEX_OPTIONS)
        else:
            q, k, v = (t.transpose(1, 2) for t in (q, xk, v))
            readable = self._decode_memory(k, v, pos)
            if readable is None:
                return torch.zeros_like(x), cache
            k, v, key_ok = readable
            if key_ok is None:
                out = flex_attention(q, k, v, block_mask=None, scale=1, kernel_options=_FLEX_OPTIONS)
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_ok.view(bsz, 1, 1, -1), scale=1)
                # A row that may read nothing (SDPA gives NaN there) contributes zero.
                out = torch.where(key_ok.any(dim=1).view(bsz, 1, 1, 1), out, torch.zeros_like(out))

        out = out * torch.exp(torch.clamp(self.value_scale, max=_EXP_CLIP)).view(1, H, 1, 1)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, H, d)
        if gate is not None:
            out = out * gate.view(bsz, seq_len, H, 1)
        return self.proj_out(out.reshape(bsz, seq_len, H * d)), cache

    def _reset_scalars(self) -> None:
        nn.init.uniform_(self.value_interp, 0, 1)
        nn.init.constant_(self.bandwidth_bias, 1.5)
        nn.init.constant_(self.bandwidth_scale, 1.5)
        nn.init.constant_(self.bandwidth_exponent, 0.33333)
        nn.init.constant_(self.value_scale, 0)

    def init_weights(self, init_std: float | None = None, factor: float = 1.0) -> None:
        init_linear([self.proj_kf, self.proj_v, self.proj_out], init_std, self.dim, factor)
        self._reset_scalars()
