# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""MosaicBlock: up to three memories (short, long, cluster) and a persistent memory (FFN).

How the memories attach to the residual stream is `layout`:

    flat                h = norm(x);  x += short(h) + long(h) + cluster(h)
    short_flat          x += short(norm(x));  h = norm(x);  x += long(h) + cluster(h)
    short_long_cluster  x += short(norm(x));  x += long(norm(x));  x += cluster(norm(x))

followed by x += ffn(norm(x)) in every layout. A memory with no heads is absent.

Each memory is a MosaicsMemory, an Attention or a DeltaNet (`<scope>_memory_type`); all
take (x, mask, positions, cache) and return (output, cache), so the block does not care
which. This is how one block covers Memory Mosaics, MMv2, MM+, two-stage MM+ and their
attention counterparts Transformer+ and two-stage Transformer+.
"""

from dataclasses import dataclass

import torch
from torch import nn

from model.common.attention import Attention
from model.common.block import feed_forward
from model.common.deltanet import DeltaNet
from model.common.feed_forward import chunked_ffn_residual
from model.common.norm import RMSNorm
from model.common.positions import Positions
from model.mosaics.config import MosaicsConfig
from model.mosaics.memory import MosaicsMemory


def build_memory(scope: str, cfg: MosaicsConfig) -> nn.Module | None:
    """The `scope` memory of a block, or None if it has no heads."""
    n_heads = cfg.heads(scope)
    if n_heads == 0:
        return None
    kind = cfg.memory_type(scope)
    if kind == "mosaics":
        return MosaicsMemory(scope, cfg)
    if kind == "attention":
        return Attention(scope, cfg.dim, cfg.head_dim, n_heads,
                         window=cfg.short_mem_window_size, skip=cfg.long_mem_skip_size,
                         cluster_mem_skip=cfg.cluster_mem_skip, rope_theta=cfg.rope_theta(scope),
                         memory_gate=cfg.memory_gate, max_seq_len=cfg.max_seq_len)
    return DeltaNet(cfg.dim, cfg.head_dim, n_heads, memory_gate=cfg.memory_gate)


@dataclass
class LayerCache:
    """Generation state of one block, per memory (a KV cache, or a DeltaNet state)."""
    long: object = None
    short: object = None
    cluster: object = None


class MosaicBlock(nn.Module):
    def __init__(self, args: MosaicsConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.layout = args.layout

        self.short_memory = build_memory("short", args)
        self.long_memory = build_memory("long", args)
        self.cluster_memory = build_memory("cluster", args)
        # Skip the cluster memory in the forward pass (kept for fine-tuning on single
        # documents, where it has nothing to read).
        self.disable_cluster = args.disable_cluster

        # One norm per input the layout needs, created only if something reads it.
        def norm():
            return RMSNorm(args.dim, eps=args.norm_eps)
        has_long, has_cluster = self.long_memory is not None, self.cluster_memory is not None
        if self.layout == "flat":
            self.attention_norm = norm()
        else:
            self.short_norm = norm() if self.short_memory is not None else None
            if self.layout == "short_flat":
                self.attention_norm = norm() if (has_long or has_cluster) else None
            else:
                self.attention_norm = norm() if has_long else None
                self.cluster_norm = norm() if has_cluster else None

        self.feed_forward = feed_forward(args)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def init_cache(self, bsz: int, max_seq_len: int, dtype, device) -> LayerCache:
        return LayerCache(**{scope: getattr(self, f"{scope}_memory").init_cache(bsz, max_seq_len, dtype, device)
                             for scope in ("short", "long", "cluster")
                             if getattr(self, f"{scope}_memory") is not None})

    @staticmethod
    def _read(memory, h, mask, pos, cache):
        """(output, cache) of one memory; output 0 if the memory is absent."""
        if memory is None:
            return 0, cache
        return memory(h, mask, pos, cache)

    def forward(self, x: torch.Tensor, masks: dict | None, pos: Positions,
                cache: LayerCache | None = None) -> tuple[torch.Tensor, LayerCache]:
        cache = cache or LayerCache()
        masks = masks or {}
        cluster = None if self.disable_cluster else self.cluster_memory
        c_short, c_long, c_cluster = cache.short, cache.long, cache.cluster

        if self.layout == "flat":
            h = self.attention_norm(x)
            short, c_short = self._read(self.short_memory, h, masks.get("short"), pos, c_short)
            long, c_long = self._read(self.long_memory, h, masks.get("long"), pos, c_long)
            clus, c_cluster = self._read(cluster, h, masks.get("cluster"), pos, c_cluster)
            x = x + long + short + clus
        else:
            if self.short_memory is not None:
                short, c_short = self._read(self.short_memory, self.short_norm(x),
                                            masks.get("short"), pos, c_short)
                x = x + short
            if self.layout == "short_flat":
                if self.attention_norm is not None:
                    h = self.attention_norm(x)
                    long, c_long = self._read(self.long_memory, h, masks.get("long"), pos, c_long)
                    clus, c_cluster = self._read(cluster, h, masks.get("cluster"), pos, c_cluster)
                    x = x + long + clus
            else:
                if self.long_memory is not None:
                    long, c_long = self._read(self.long_memory, self.attention_norm(x),
                                              masks.get("long"), pos, c_long)
                    x = x + long
                if cluster is not None:
                    clus, c_cluster = self._read(cluster, self.cluster_norm(x),
                                                 masks.get("cluster"), pos, c_cluster)
                    x = x + clus

        out = chunked_ffn_residual(x, self.ffn_norm, self.feed_forward)
        return out, LayerCache(long=c_long, short=c_short, cluster=c_cluster)

    def init_weights(self, init_std: float | None = None, factor: float = 1.0) -> None:
        for memory in (self.long_memory, self.short_memory, self.cluster_memory):
            if memory is not None:
                memory.init_weights(init_std, factor)
        for norm in (getattr(self, n, None) for n in ("short_norm", "attention_norm", "cluster_norm")):
            if norm is not None:
                norm.reset_parameters()
        self.feed_forward.init_weights(init_std, factor)
        self.ffn_norm.reset_parameters()
