# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Weight initialization shared by all model families."""

from torch import nn


def layer_init_factor(method: str, layer_depth: int, n_layers: int, dim: int) -> float:
    """Divisor of layer `layer_depth`'s init std (see BaseConfig.init_method)."""
    return {
        "current_depth": (2 * (layer_depth + 1)) ** 0.5,
        "global_depth": (2 * (n_layers + 1)) ** 0.5,
        "dim_ratio": dim / 4096,
        "none": 1.0,
    }[method]


def init_linear(linears, init_std: float | None, dim: int, factor: float = 1.0) -> None:
    """Truncated-normal init, std = (init_std or dim^-0.5) / factor, cut at 3 std."""
    std = (init_std or dim ** -0.5) / factor
    for w in linears:
        nn.init.trunc_normal_(w.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
