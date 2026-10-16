# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Diagnostic probes: per-component gradient and update norms.

Splits parameters into components by name (embedding / short / long / cluster / ffn /
norm / output) and reports the gradient norm of each, plus the relative size of the
AdamW step actually taken. Useful for spotting a memory that is not learning, or a
component whose updates are orders of magnitude off the rest.

Off the hot path: the train loop only calls these on logging steps, and
`compute_component_update_norms` needs a pre-step parameter snapshot, so it is only
taken when a log step is due.
"""

import torch
from torch.distributed._tensor import DTensor

def _grad_component(name: str) -> str:
    """Classify a parameter (by name) into a memory component for grad-norm logging."""
    if "cluster_memory" in name:
        return "cluster"
    if "long_memory" in name:
        return "long"
    if "short_memory" in name:
        return "short"
    if ".mixer." in name:              # Transformer / RNN layers
        return "mixer"
    if "feed_forward" in name:
        return "ffn"
    if "tok_embeddings" in name or "output" in name:
        return "embed"
    return "other"

def compute_component_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    """Raw (pre-clip) global grad L2 norm per memory component.

    Uses clip_grad_norm_ with a huge max_norm: it returns the correct
    FSDP/DTensor mesh-reduced norm and is non-destructive (clip coefficient
    clamps to 1.0, so gradients are left unchanged). Call BEFORE the real clip.
    """
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for name, p in model.named_parameters():
        if p.grad is not None:
            groups[_grad_component(name)].append(p)
    norms: dict[str, float] = {}
    for comp, params in groups.items():
        gn = torch.nn.utils.clip_grad_norm_(params, max_norm=1e30)
        norms[comp] = gn.full_tensor().item() if isinstance(gn, DTensor) else float(gn)
    return norms

def compute_component_update_norms(model: torch.nn.Module, prev_params: dict) -> dict[str, float]:
    """Per-component update norm ‖Δw‖ for the last optimizer step.

    Δw = w_now - prev (the actual AdamW step, incl. weight decay), captured by
    snapshotting params before optimizer.step(). Unlike the raw grad norm, this
    reflects what Adam actually applied. Sums-of-squares accumulate per component
    (DTensor partials add cleanly), then are mesh-reduced once per component.
    """
    sq_dw: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if name not in prev_params:
            continue
        comp = _grad_component(name)
        sdw = ((p.detach() - prev_params[name]).to(torch.float32) ** 2).sum()
        sq_dw[comp] = sdw if comp not in sq_dw else sq_dw[comp] + sdw
    out: dict[str, float] = {}
    for comp, s in sq_dw.items():
        if isinstance(s, DTensor):
            s = s.full_tensor()
        out[comp] = float(s.sqrt().item())
    return out
