# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Configuration of the Memory Mosaics family (`model_type: mosaics`).

A block has up to three memories, each implemented by one of:

    "mosaics"    the Mosaics memory: the query is the key times a bandwidth b1 * n^a + b0
                 growing with the memory size n, values look one token ahead (v_t holds
                 x_{t+1}), no position encoding; it never reads its own position
    "attention"  softmax attention: independent q/k/v, plain values, optional RoPE
    "deltanet"   gated DeltaNet, a fixed-size recurrent state (short memory only)

and each Mosaics memory's key is a leaky average over the past, k_T = phi(x_T, x_{T-1},
...), or the current token alone, k_T = phi(x_T) (`<scope>_key_feature`). Settings that
do not apply to a memory's type are rejected.
"""

from dataclasses import dataclass

from model.common.config import BaseConfig

MEMORY_TYPES = {"short": ("mosaics", "attention", "deltanet"),
                "long": ("mosaics", "attention"),
                "cluster": ("mosaics", "attention")}
KEY_FEATURES = ("leaky_average", "plain")
LAYOUTS = ("flat", "short_flat", "short_long_cluster")


@dataclass
class MosaicsConfig(BaseConfig):
    short_memory_type: str = "mosaics"
    long_memory_type: str = "mosaics"
    cluster_memory_type: str = "mosaics"

    # Heads per memory; 0 removes it. None = dim // head_dim (short, long), 0 (cluster).
    n_short_heads: int | None = None
    n_long_heads: int | None = None
    n_cluster_heads: int | None = None

    # Mosaics memories: "leaky_average" | "plain" (None = leaky_average).
    short_key_feature: str | None = None
    long_key_feature: str | None = None
    cluster_key_feature: str | None = None
    # In training the leaky average is computed in chunks of ~recurrence_chunk_size
    # tokens, each warmed up on the previous recurrence_padding; generation runs it whole.
    recurrence_padding: int = 256
    recurrence_chunk_size: int = 1024

    # Attention memories: "rope" | "nope" (None = rope for short, nope for long/cluster),
    # and the RoPE base (None = 10000).
    short_mem_position_encoding: str | None = None
    long_mem_position_encoding: str | None = None
    cluster_mem_position_encoding: str | None = None
    short_mem_rope_theta: float | None = None
    long_mem_rope_theta: float | None = None
    cluster_mem_rope_theta: float | None = None

    # What each memory reads (attn_bias_type decides whether documents are separate):
    short_mem_window_size: int = 128      # short: the last N tokens (0 = all); 0 for deltanet
    long_mem_skip_size: int = 128         # long: everything at least N tokens back
    random_long_mem_skip_size: int = 0    # in training, add U{0..N} to the skip each step
    cluster_mem_skip: bool = True         # cluster: earlier documents only (else the whole row)
    disable_cluster: bool = False         # skip the cluster memory (fine-tuning on single documents)

    # How the memories attach to the residual stream (see model.mosaics.block).
    layout: str = "short_long_cluster"

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.layout in LAYOUTS, f"layout={self.layout!r} not in {LAYOUTS}"
        for scope, types in MEMORY_TYPES.items():
            kind = self.memory_type(scope)
            assert kind in types, f"{scope}_memory_type={kind!r} not in {types}"
            key = getattr(self, f"{scope}_key_feature")
            pe = getattr(self, f"{scope}_mem_position_encoding")
            theta = getattr(self, f"{scope}_mem_rope_theta")
            if kind != "mosaics":
                assert key is None, f"{scope}_key_feature applies to a mosaics memory, not {kind!r}"
            else:
                assert key in (None,) + KEY_FEATURES, f"{scope}_key_feature={key!r} not in {KEY_FEATURES}"
            if kind != "attention":
                assert pe is None and theta is None, \
                    f"{scope}_mem_position_encoding / _rope_theta apply to an attention memory, not {kind!r}"
            else:
                assert pe in (None, "rope", "nope"), f"{scope}_mem_position_encoding={pe!r}"
                assert theta is None or self.rope_theta(scope) is not None, \
                    f"{scope}_mem_rope_theta is set but the {scope} memory has no RoPE"
        if self.long_memory_type == "mosaics" and self.heads("long"):
            # Its value v_t holds x_{t+1}: with skip 0 a query would read its own target.
            assert self.long_mem_skip_size >= 1, "a mosaics long memory needs long_mem_skip_size >= 1"
        if self.short_memory_type == "deltanet":
            assert self.short_mem_window_size == 0, "a deltanet short memory has no window: set short_mem_window_size: 0"
        assert not (self.heads("cluster") and not self.documentwise), \
            "cluster memory reads earlier documents: it needs attn_bias_type: doc_causal"

    def memory_type(self, scope: str) -> str:
        return getattr(self, f"{scope}_memory_type")

    def heads(self, scope: str) -> int:
        n = getattr(self, f"n_{scope}_heads")
        if n is None:
            return 0 if scope == "cluster" else self.dim // self.head_dim
        return n

    def leaky_key(self, scope: str) -> bool:
        return (getattr(self, f"{scope}_key_feature") or "leaky_average") == "leaky_average"

    def rope_theta(self, scope: str) -> float | None:
        """RoPE base of an attention memory, None if it has no position encoding."""
        pe = getattr(self, f"{scope}_mem_position_encoding") or ("rope" if scope == "short" else "nope")
        if pe == "nope":
            return None
        return getattr(self, f"{scope}_mem_rope_theta") or 10000.0
