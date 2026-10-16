# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Optimizer and LR scheduler construction."""

import math
from collections.abc import Callable
from functools import partial

import torch
from torch.optim import AdamW, lr_scheduler

from training.config import OptimConfig


def lr_cosine(step: int, *, warmup: int, n_steps: int, cycle_length: float, theta: float, min_ratio: float) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    t = (step - warmup) / max(n_steps - warmup, 1) / cycle_length
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * t**theta))


def lr_linear(step: int, *, warmup: int, n_steps: int, min_ratio: float) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    t = (step - warmup) / max(n_steps - warmup, 1)
    return min_ratio + (1 - min_ratio) * (1 - t)


def lr_inv_sqrt(step: int, *, warmup: int, exp_factor: float, min_ratio: float) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    return max(min_ratio, (warmup / step) ** exp_factor)


def build_lr_fn(args: OptimConfig, n_steps: int) -> Callable:
    if args.scheduler == "constant":
        return lambda step: 1.0
    elif args.scheduler == "cosine":
        return partial(
            lr_cosine, warmup=args.warmup, n_steps=n_steps,
            cycle_length=args.cycle_length, theta=args.cosine_theta, min_ratio=args.lr_min_ratio,
        )
    elif args.scheduler == "linear":
        return partial(lr_linear, warmup=args.warmup, n_steps=n_steps, min_ratio=args.lr_min_ratio)
    elif args.scheduler == "inv_sqrt":
        return partial(lr_inv_sqrt, warmup=args.warmup, exp_factor=args.exp_factor, min_ratio=args.lr_min_ratio)
    else:
        raise NotImplementedError(f"Unknown scheduler: {args.scheduler}")


def _is_no_decay(name: str) -> bool:
    """Params excluded from weight decay: the bandwidth coefficients of the mosaics memories
    (bandwidth_bias / _scale / _exponent), learned values whose optimum is not 0."""
    return "bandwidth" in name


def build_optimizer(
    model: torch.nn.Module, args: OptimConfig, n_steps: int,
) -> tuple[torch.optim.Optimizer, lr_scheduler.LRScheduler]:
    # Zero-element params (e.g. a disabled memory's projections when n_*_heads == 0) are
    # skipped entirely: they never get a gradient, so AdamW makes no state and the saved
    # checkpoint has none -- including them makes resume expect a missing ".step".
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and p.numel() > 0]

    # Two groups: bandwidth params get weight_decay=0 (decaying a bandwidth coefficient
    # pulls it toward a wrong 0 and -- for a barely-moving param -- is most of the gradient
    # it sees). Everything else keeps args.weight_decay (norms and biases included).
    decay = [p for n, p in named if not _is_no_decay(n)]
    no_decay = [p for n, p in named if _is_no_decay(n)]
    groups = [g for g in (                          # drop an empty group
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ) if g["params"]]

    optimizer = AdamW(
        groups,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.epsilon,
        fused=args.fused_optimizer,
    )
    lr_fn = build_lr_fn(args, n_steps)
    sched = lr_scheduler.LambdaLR(optimizer, lr_fn)     # single lambda -> applied to all groups
    return optimizer, sched
