# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""The Transformer family: the Transformer baseline and the hybrids.

Every layer is a MixerBlock (norm -> mixer -> residual -> norm -> FFN) whose mixer is

    full      causal softmax attention (RoPE or no position encoding)
    swa       sliding-window attention with RoPE        (Hybrid SWA: even layers)
    deltanet  gated DeltaNet                             (Hybrid DeltaNet: even layers)

The mixers are the shared layers of model.common, so a Transformer computes exactly what
the corresponding attention memory of a MosaicBlock would.
"""

from model.common.attention import Attention
from model.common.block import MixerBlock
from model.common.deltanet import DeltaNet
from model.common.lm import LanguageModel
from model.common.mask import memory_mask
from model.transformer.config import TransformerConfig


def _layer(args: TransformerConfig, layer_id: int) -> MixerBlock:
    kind, H = args.layer_kind(layer_id), args.heads
    if kind == "deltanet":
        return MixerBlock(DeltaNet(args.dim, args.head_dim, H, memory_gate=args.memory_gate), args)
    if kind == "swa":
        mixer = Attention("short", args.dim, args.head_dim, H, window=args.swa_window,
                          rope_theta=args.swa_rope_theta, memory_gate=args.memory_gate,
                          max_seq_len=args.max_seq_len)
    else:
        mixer = Attention("long", args.dim, args.head_dim, H, skip=0,
                          rope_theta=args.rope_theta if args.position_encoding == "rope" else None,
                          memory_gate=args.memory_gate, max_seq_len=args.max_seq_len)
    return MixerBlock(mixer, args, mask_key=kind)


class Transformer(LanguageModel):
    def __init__(self, args: TransformerConfig) -> None:
        super().__init__(args, [_layer(args, i) for i in range(args.n_layers)])

    def build_masks(self, tokens, q_lens, kv_lens, attn_bias_type, step, seed) -> dict:
        kinds = {self.args.layer_kind(i) for i in range(self.n_layers)}
        masks = {"full": memory_mask("long", True, tokens, q_lens, attn_bias_type, kv_lens=kv_lens, skip=0)}
        if "swa" in kinds:
            masks["swa"] = memory_mask("short", True, tokens, q_lens, attn_bias_type,
                                       kv_lens=kv_lens, window=self.args.swa_window)
        return masks
