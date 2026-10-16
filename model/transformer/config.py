# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Configuration of the Transformer family (`model_type: transformer`): the Transformer
and the hybrids Hybrid SWA and Hybrid DeltaNet."""

from dataclasses import dataclass

from model.common.config import BaseConfig

HYBRIDS = (None, "swa", "deltanet")


@dataclass
class TransformerConfig(BaseConfig):
    n_heads: int | None = None            # None = dim // head_dim
    # Full-attention layers: "rope" | "nope", and the RoPE base.
    position_encoding: str = "rope"
    rope_theta: float = 10000.0
    # Hybrids interleave 1:1: even layers (0, 2, ...) are local -- sliding-window
    # attention ("swa") or gated DeltaNet ("deltanet") -- and odd layers full attention.
    hybrid: str | None = None
    swa_window: int = 1024
    swa_rope_theta: float = 250000.0      # SWA layers always use RoPE

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.position_encoding in ("rope", "nope"), f"position_encoding={self.position_encoding!r}"
        assert self.hybrid in HYBRIDS, f"hybrid={self.hybrid!r} not in {HYBRIDS}"

    @property
    def heads(self) -> int:
        return self.n_heads if self.n_heads is not None else self.dim // self.head_dim

    def layer_kind(self, layer_id: int) -> str:
        """"full" | "swa" | "deltanet"."""
        return self.hybrid if (self.hybrid and layer_id % 2 == 0) else "full"
