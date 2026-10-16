# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Gated DeltaNet: a recurrent memory with a fixed-size state.

Per head, a state S (d x d) is updated by the gated delta rule and read by the query:

    S_t = exp(g_t) S_{t-1} + beta_t k_t (v_t - exp(g_t) S_{t-1}^T k_t)^T,   o_t = d^-1/2 S_t^T q_t

with ||k_t|| = 1, write strength beta_t in [0, 1] and a data-dependent log-decay g_t <= 0
(g_t = -softplus(.) * exp(A_log), Mamba-2 / GatedDeltaNet style). This is the [K, V]
orientation flash-linear-attention uses. The state restarts at every document under
"doc_causal" (Positions.doc_lens), otherwise at every row. There is no window: a delta
rule cannot evict the token leaving one.

On CUDA the recurrence runs in fla's chunked kernel (needs flash-linear-attention);
elsewhere in `reference_delta_rule`, a plain loop that is exact and slow and is the
specification the kernel is tested against. Generation carries the state (DeltaNetCache)
instead of a KV cache: a prefill that returns the final state, then one exact recurrent
step per token.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from model.common.init import init_linear
from model.common.mask import FlexAttentionBias
from model.common.positions import Positions

_EXP_CLIP = 15.0
# CUDA's limit on launch grid axis y; fla's state kernels put (segment, head) there.
_MAX_GRID_Y = 65535
# Use the PyTorch reference on CUDA too, e.g. to test logic exactly.
FORCE_REFERENCE = False


@dataclass
class DeltaNetCache:
    """Generation state: S per row and head, [bsz, H, d, d] in fp32 (S is rescaled by
    exp(g) every step, which drifts in bf16). Its size does not grow with the sequence."""
    state: torch.Tensor
    pos: int = 0

    def reset_rows(self, rows: torch.Tensor) -> None:
        """Zero the state of rows flagged in `rows` (bool [bsz]): each starts a new
        document, as the document-wise training recurrence does after an EOS."""
        self.state[rows] = 0


def _delta_step(S, q, k, v, beta, g, scale):
    """One token: S [N, H, d, d], q/k/v [N, H, d], beta/g [N, H] -> (o [N, H, d], S)."""
    S = S * g.exp()[..., None, None]
    S = S + beta[..., None, None] * k.unsqueeze(-1) * (v - torch.einsum("nhkv,nhk->nhv", S, k)).unsqueeze(-2)
    return torch.einsum("nhkv,nhk->nhv", S, q * scale), S


def reference_delta_rule(q, k, v, g, beta, cu_seqlens, scale, initial_state=None,
                         output_final_state=False):
    """The gated delta rule in plain PyTorch, with fla's varlen interface: inputs
    [1, T, H, d] (beta, g: [1, T, H]) holding segments cut at cu_seqlens; one state per
    segment. Returns (outputs [1, T, H, d], final states [S, H, d, d] or None)."""
    H, d = q.shape[2], q.shape[3]
    bounds = cu_seqlens.tolist()
    out = torch.empty_like(v, dtype=torch.float32)
    finals = []
    for s in range(len(bounds) - 1):
        S = (initial_state[s:s + 1].float() if initial_state is not None
             else q.new_zeros(1, H, d, d, dtype=torch.float32))
        for t in range(bounds[s], bounds[s + 1]):
            o, S = _delta_step(S, q[:, t].float(), k[:, t].float(), v[:, t].float(),
                               beta[:, t].float(), g[:, t].float(), scale)
            out[:, t] = o
        finals.append(S)
    return out.to(v.dtype), (torch.cat(finals) if output_final_state else None)


class DeltaNet(nn.Module):
    see_self = True
    uses_mask = False         # its scope comes from Positions.doc_lens

    def __init__(self, dim: int, head_dim: int, n_heads: int, *, memory_gate: bool = False) -> None:
        super().__init__()
        assert head_dim <= 256, "flash-linear-attention supports head_dim <= 256"
        self.dim, self.head_dim, self.n_heads = dim, head_dim, n_heads
        self.memory_gate = memory_gate
        self.scale = head_dim ** -0.5
        H, d = n_heads, head_dim
        self.proj_q = nn.Linear(dim, H * d, bias=False)
        self.proj_k = nn.Linear(dim, H * d, bias=False)
        self.proj_v = nn.Linear(dim, H * d, bias=False)
        self.proj_out = nn.Linear(H * d, dim, bias=False)
        self.proj_beta = nn.Linear(dim, H, bias=False)    # write strength
        self.proj_decay = nn.Linear(dim, H, bias=False)   # data-dependent log-decay
        self.A_log = nn.Parameter(torch.zeros(H))         # per-head decay range
        self.dt_bias = nn.Parameter(torch.zeros(H))
        self.value_scale = nn.Parameter(torch.zeros(H))   # log output gain
        if memory_gate:
            self.proj_gate = nn.Linear(dim, H, bias=False)

    def init_cache(self, bsz: int, max_seq_len: int, dtype, device) -> DeltaNetCache:
        """The generation state; sized by the head geometry alone, kept in fp32."""
        d = self.head_dim
        return DeltaNetCache(torch.zeros(bsz, self.n_heads, d, d, dtype=torch.float32, device=device))

    # ── the recurrence ──

    @torch.compiler.disable(recursive=True)
    def _delta(self, q, k, v, g, beta, cu_seqlens, initial_state=None, output_final_state=False):
        """The gated delta rule over packed segments; returns (outputs, final states).
        Excluded from torch.compile (triton kernel, data-dependent segment boundaries),
        so blocks holding a DeltaNet compile with fullgraph=False."""
        if not q.is_cuda or FORCE_REFERENCE:
            return reference_delta_rule(q, k, v, g, beta, cu_seqlens, self.scale,
                                        initial_state, output_final_state)
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        except ImportError as e:
            raise ImportError("DeltaNet on CUDA needs flash-linear-attention (pip install "
                              "flash-linear-attention)") from e

        def fla(lo, hi, cu, init):
            return chunk_gated_delta_rule(
                q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], g[:, lo:hi], beta[:, lo:hi],
                scale=self.scale, initial_state=init,
                output_final_state=output_final_state, cu_seqlens=cu)

        # fla launches one program per (segment, head) on grid axis y, capped at 65535,
        # e.g. exceeded by SFT rows where every pad token (pad == EOS under a Mistral
        # tokenizer) is its own document. Segments are independent: run whole groups.
        n_seg, per_call = cu_seqlens.numel() - 1, _MAX_GRID_Y // q.shape[2]
        if n_seg <= per_call:
            return fla(0, q.shape[1], cu_seqlens, initial_state)
        bounds = cu_seqlens.tolist()
        outs, states = [], []
        for a in range(0, n_seg, per_call):
            b = min(a + per_call, n_seg)
            o, S = fla(bounds[a], bounds[b], cu_seqlens[a:b + 1] - bounds[a],
                       None if initial_state is None else initial_state[a:b])
            outs.append(o)
            states.append(S)
        return torch.cat(outs, dim=1), (torch.cat(states) if output_final_state else None)

    @staticmethod
    def _cu_seqlens(lens: torch.Tensor) -> torch.Tensor:
        cu = lens.new_zeros(lens.numel() + 1, dtype=torch.int32)
        cu[1:] = torch.cumsum(lens.to(torch.int32), dim=0)
        return cu

    def _full_sequence(self, q, k, v, beta, g, pos: Positions, cache: DeltaNetCache | None):
        """All tokens at once (training, evaluation, prefill). With a cache, also returns
        each row's final state: its last document's, which decoding continues."""
        bsz, seq_len, H, d = q.shape
        # One recurrence per document, or per row (rows never share state). fla's varlen
        # API takes batch 1 + segment offsets; rows and documents are both row-major.
        lens = pos.doc_lens if pos.doc_lens is not None else \
            torch.full((bsz,), seq_len, device=q.device, dtype=torch.long)
        cu = self._cu_seqlens(lens)
        flat = [t.reshape(1, bsz * seq_len, *t.shape[2:]) for t in (q, k, v, g, beta)]
        if cache is None:
            o, _ = self._delta(*flat, cu)
            return o.reshape(bsz, seq_len, H, d), None
        assert cache.pos == 0, "prefill into a non-empty cache is not supported"
        # Every row ends at a segment boundary; the row's last segment is its state.
        row_ends = torch.arange(1, bsz + 1, device=cu.device, dtype=cu.dtype) * seq_len
        last_seg = torch.searchsorted(cu[1:], row_ends)
        assert torch.equal(cu[1:][last_seg], row_ends), "a document crosses a row boundary"
        o, S = self._delta(*flat, cu, output_final_state=True)
        return o.reshape(bsz, seq_len, H, d), DeltaNetCache(S[last_seg].float().contiguous(), seq_len)

    def forward(self, x: torch.Tensor, mask: FlexAttentionBias | None, pos: Positions,
                cache: DeltaNetCache | None = None) -> tuple[torch.Tensor, DeltaNetCache | None]:
        bsz, seq_len, _ = x.shape
        H, d = self.n_heads, self.head_dim
        q = self.proj_q(x).view(bsz, seq_len, H, d)
        k = F.normalize(self.proj_k(x).view(bsz, seq_len, H, d), dim=-1)
        v = self.proj_v(x).view(bsz, seq_len, H, d)
        beta = torch.sigmoid(self.proj_beta(x))                                   # (B, T, H)
        g = -F.softplus(self.proj_decay(x).float() + self.dt_bias.view(1, 1, H)) \
            * torch.exp(self.A_log).view(1, 1, H)                                 # log-decay, fp32

        if cache is not None and cache.pos > 0:      # a decoding step
            assert seq_len == 1, "decode one token at a time"
            o, S = _delta_step(cache.state, q[:, 0].float(), k[:, 0].float(), v[:, 0].float(),
                               beta[:, 0].float(), g[:, 0].float(), self.scale)
            out, cache = o.unsqueeze(1).to(q.dtype), DeltaNetCache(S, cache.pos + 1)
        else:
            out, cache = self._full_sequence(q, k, v, beta, g, pos, cache)

        out = out * torch.exp(torch.clamp(self.value_scale, max=_EXP_CLIP)).view(1, 1, H, 1)
        if self.memory_gate:
            out = out * torch.sigmoid(self.proj_gate(x)).view(bsz, seq_len, H, 1)
        return self.proj_out(out.reshape(bsz, seq_len, H * d)), cache

    def init_weights(self, init_std: float | None = None, factor: float = 1.0) -> None:
        projs = [self.proj_q, self.proj_k, self.proj_v, self.proj_out, self.proj_beta, self.proj_decay]
        if self.memory_gate:
            projs.append(self.proj_gate)
        init_linear(projs, init_std, self.dim, factor)
        nn.init.constant_(self.value_scale, 0)
        nn.init.constant_(self.dt_bias, 0.0)
        # A = exp(A_log) ~ Uniform(1, 16): the state forgets slowly at init (GDN convention).
        nn.init.uniform_(self.A_log, 1.0, 16.0)
        with torch.no_grad():
            self.A_log.log_()
