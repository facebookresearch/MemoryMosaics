# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""LeakyAverage: real-valued leaky average recurrence with custom CUDA kernel.

Recurrence: y_t = omega_t * y_{t-1} + u_t

Pre-processing (torch.compile friendly):
  omega = exp(-|decay_logits|)           decay in (0, 1]
  gate = exp(clamp(gate_logits, max))    gate > 0, bounded
  gate expansion: compact (C_wk) -> full (C)
  u = gate * x

Post-processing (torch.compile friendly):
  F.normalize(y, dim=-1) / sqrt(n_decays_per_head)

On CUDA the scan runs in a custom kernel (cuda/gatesum_slim, JIT-built on first use, bf16).
Elsewhere it falls back to `leaky_avg_reference`, a plain PyTorch loop over time: slow, but
exact, differentiable, and the specification the kernel is tested against.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_cuda_module = None
# Use the PyTorch reference on CUDA too, e.g. to test logic exactly (the kernel takes bf16).
FORCE_REFERENCE = False


def _kernels():
    """The CUDA kernels, JIT-built on first call. Called only inside the custom ops'
    implementations, which torch.compile never traces (it uses their fake versions), so
    a compiled model can build the kernels at its first run."""
    global _cuda_module
    if _cuda_module is None:
        rootdir = os.path.join(os.path.dirname(__file__), "cuda", "gatesum_slim")
        _cuda_module = load(
            name="gatesum_slim_cuda_bf16",
            sources=[
                os.path.join(rootdir, "gatesum_op_bf16.cpp"),
                os.path.join(rootdir, "gatesum_cuda_bf16.cu"),
            ],
            verbose=True,
            extra_cuda_cflags=[
                "-t 4", "-std=c++20", "-res-usage",
                "--maxrregcount 60", "--use_fast_math",
                "-O3", "-Xptxas -O3", "--extra-device-vectorization","-allow-unsupported-compiler"
            ],
        )
    return _cuda_module


def leaky_avg_reference(
    omega: torch.Tensor, u: torch.Tensor, c_wk: int, y_prev: torch.Tensor | None = None,
) -> torch.Tensor:
    """y_t = omega_t * y_{t-1} + u_t, with y_{-1} = y_prev (default 0).

    omega: (B, T, c_wk), one decay per group of C // c_wk consecutive channels.
    u:     (B, T, C).
    """
    B, T, C = u.shape
    omega = omega.repeat_interleave(C // c_wk, dim=-1)
    y = u.new_zeros(B, C) if y_prev is None else y_prev.to(u.dtype)
    out = []
    for t in range(T):
        y = omega[:, t] * y + u[:, t]
        out.append(y)
    return torch.stack(out, dim=1)


# ── custom_op registration (graph-break-free for torch.compile) ──

@torch.library.custom_op("mosaics::leaky_avg_fwd", mutates_args=(), device_types="cuda")
def _leaky_avg_fwd(omega: torch.Tensor, u: torch.Tensor, c_wk: int) -> torch.Tensor:
    B, T, C = u.shape
    assert B * C % min(C, 32) == 0
    omega = omega.to(torch.bfloat16).contiguous()
    u = u.to(torch.bfloat16).contiguous()
    y = torch.empty((B, T, C), device=omega.device, dtype=torch.bfloat16)
    _kernels().forward(B, T, C, c_wk, omega, u, y)
    return y


@torch.library.register_fake("mosaics::leaky_avg_fwd")
def _leaky_avg_fwd_fake(omega: torch.Tensor, u: torch.Tensor, c_wk: int) -> torch.Tensor:
    B, T, C = u.shape
    return u.new_empty((B, T, C), dtype=torch.bfloat16)


@torch.library.custom_op("mosaics::leaky_avg_bwd", mutates_args=(), device_types="cuda")
def _leaky_avg_bwd(
    omega: torch.Tensor, y: torch.Tensor, gy: torch.Tensor, c_wk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, C = gy.shape
    gomega = torch.zeros((B, T, c_wk), device=gy.device, dtype=torch.float32)
    gu = torch.empty((B, T, C), device=gy.device, dtype=torch.bfloat16)
    _kernels().backward(B, T, C, c_wk, omega, y, gy.contiguous(), gomega, gu)
    return gomega, gu


@torch.library.register_fake("mosaics::leaky_avg_bwd")
def _leaky_avg_bwd_fake(
    omega: torch.Tensor, y: torch.Tensor, gy: torch.Tensor, c_wk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, C = gy.shape
    return (
        gy.new_empty((B, T, c_wk), dtype=torch.float32),
        gy.new_empty((B, T, C), dtype=torch.bfloat16),
    )


def _leaky_avg_setup_context(ctx, inputs, output):
    omega, u, c_wk = inputs
    ctx.save_for_backward(omega, output)
    ctx.c_wk = c_wk


def _leaky_avg_backward(ctx, gy):
    omega, y = ctx.saved_tensors
    gomega, gu = _leaky_avg_bwd(omega, y, gy, ctx.c_wk)
    return gomega, gu, None


torch.library.register_autograd(
    "mosaics::leaky_avg_fwd",
    _leaky_avg_backward,
    setup_context=_leaky_avg_setup_context,
)


# State-padded prefill (generation only, inference_mode → no autograd needed).
# Registered as a custom op + fake so fullgraph=True torch.compile can trace it
# without a graph break (the @torch.compiler.disable approach forbade fullgraph).
@torch.library.custom_op("mosaics::leaky_avg_state_pad", mutates_args=(), device_types="cuda")
def _leaky_avg_state_pad(
    omega: torch.Tensor, u: torch.Tensor, loc: torch.Tensor, c_wk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, C = u.shape
    omega = omega.to(torch.bfloat16).contiguous()
    u = u.to(torch.bfloat16).contiguous()
    y = torch.zeros((B, T, C), device=omega.device, dtype=torch.bfloat16)
    y_state = torch.zeros((B, C), device=omega.device, dtype=torch.float32)
    _kernels().forward_state_pad(
        B, T, C, c_wk, omega, u, y, loc.to(torch.int32).contiguous(), y_state,
    )
    return y, y_state


@torch.library.register_fake("mosaics::leaky_avg_state_pad")
def _leaky_avg_state_pad_fake(
    omega: torch.Tensor, u: torch.Tensor, loc: torch.Tensor, c_wk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, C = u.shape
    return (
        u.new_empty((B, T, C), dtype=torch.bfloat16),
        u.new_empty((B, C), dtype=torch.float32),
    )


# Single-step cached decode (generation only). The CUDA kernel reads the prior
# recurrence state from y_prev and writes the updated state back into it, so
# y_prev is mutated IN PLACE (mutates_args) and must NOT be copied/made
# contiguous here. Registered as a custom op so fullgraph=True can trace it.
@torch.library.custom_op("mosaics::leaky_avg_forward_cache", mutates_args=("y_prev",), device_types="cuda")
def _leaky_avg_forward_cache(
    omega: torch.Tensor, u: torch.Tensor, y_prev: torch.Tensor, c_wk: int,
) -> torch.Tensor:
    B, T, C = u.shape
    omega = omega.to(torch.bfloat16).contiguous()
    u = u.to(torch.bfloat16).contiguous()
    y = torch.empty((B, T, C), device=omega.device, dtype=torch.bfloat16)
    _kernels().forward_cache(B, T, C, c_wk, omega, u, y, y_prev)
    return y


@torch.library.register_fake("mosaics::leaky_avg_forward_cache")
def _leaky_avg_forward_cache_fake(
    omega: torch.Tensor, u: torch.Tensor, y_prev: torch.Tensor, c_wk: int,
) -> torch.Tensor:
    B, T, C = u.shape
    return u.new_empty((B, T, C), dtype=torch.bfloat16)


def _bf16(*tensors):
    """The kernel computes in bf16 (accumulating in fp32). Casting outside the custom op
    lets autograd carry gradients through the cast, so the module returns -- and takes
    gradients in -- the caller's dtype; for bf16 inputs the casts are no-ops."""
    return [t.to(torch.bfloat16) for t in tensors]


class LeakyAverage(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_decays_per_head: int = 1,
        scale: float = 1,
        exp_clip: float = 15,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_decays_per_head = n_decays_per_head
        self.scale = scale
        self.c_wk = n_head * n_decays_per_head
        self.exp_clip = exp_clip

    def _prepare_inputs(self, x, decay_logits, gate_logits):
        omega = torch.exp(-decay_logits.abs()) * self.scale
        if gate_logits is not None:
            gate = torch.exp(torch.clamp(gate_logits, max=self.exp_clip))
            B, T, C = x.shape
            G = C // self.c_wk
            u = gate.unsqueeze(-1).expand(B, T, self.c_wk, G).reshape(B, T, C) * x
        else:
            u = x
        return omega, u

    def _post_normalize(self, y):
        B, T, C = y.shape
        head_dim = C // self.n_head
        sub_dim = head_dim // self.n_decays_per_head
        y = y.view(B, T, self.n_head, self.n_decays_per_head, sub_dim)
        y = F.normalize(y, dim=-1) / math.sqrt(self.n_decays_per_head)
        return y.view(B, T, self.n_head, head_dim)

    def _cuda_forward(self, omega, u):
        if not u.is_cuda or FORCE_REFERENCE:
            return leaky_avg_reference(omega, u, self.c_wk)
        return _leaky_avg_fwd(*_bf16(omega, u), self.c_wk).to(u.dtype)

    def _cuda_forward_cache(self, omega, u, y_prev):
        if not u.is_cuda or FORCE_REFERENCE:
            y = leaky_avg_reference(omega, u, self.c_wk, y_prev)
            y_prev.copy_(y[:, -1])
            return y
        return _leaky_avg_forward_cache(*_bf16(omega, u), y_prev, self.c_wk).to(u.dtype)

    def _cuda_forward_state_pad(self, omega, u, loc):
        if not u.is_cuda or FORCE_REFERENCE:
            y = leaky_avg_reference(omega, u, self.c_wk)
            y_state = y[torch.arange(y.shape[0], device=y.device), loc.long()].float()
            return y, y_state
        y, y_state = _leaky_avg_state_pad(*_bf16(omega, u), loc, self.c_wk)
        return y.to(u.dtype), y_state

    def forward(self, x, decay_logits, gate_logits=None, loc=None):
        assert loc is None
        omega, u = self._prepare_inputs(x, decay_logits, gate_logits)
        y = self._cuda_forward(omega, u)
        return self._post_normalize(y)

    def forward_cache(self, x, decay_logits, gate_logits, y_prev):
        omega, u = self._prepare_inputs(x, decay_logits, gate_logits)
        y = self._cuda_forward_cache(omega, u, y_prev)
        return self._post_normalize(y)

    def forward_state_pad(self, x, decay_logits, gate_logits, loc):
        omega, u = self._prepare_inputs(x, decay_logits, gate_logits)
        y, y_state = self._cuda_forward_state_pad(omega, u, loc)
        return self._post_normalize(y), y_state
