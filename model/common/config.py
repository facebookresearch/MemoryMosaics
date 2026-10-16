# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Fields every model family shares. Each family extends BaseConfig with its own."""

from dataclasses import dataclass

ATTN_BIAS_TYPES = ("causal", "doc_causal")
INIT_METHODS = ("current_depth", "global_depth", "dim_ratio", "none")


@dataclass
class BaseConfig:
    dim: int = 512
    n_layers: int = 8
    head_dim: int = 64
    max_seq_len: int = 4096               # the training length (set from seq_len)
    vocab_size: int = -1                  # set from the tokenizer

    # Persistent memory (FFN, SwiGLU): hidden = multiple_of-rounded 2/3 * ffn_exp * dim
    # (times ffn_dim_multiplier).
    ffn_dim_multiplier: float | None = None
    ffn_exp: int = 4
    multiple_of: int = 256

    norm_eps: float = 1e-5
    weight_tying: bool = False
    init_method: str = "current_depth"    # per-layer init scale, see model.common.init
    init_base_std: float | None = None    # None = dim ** -0.5
    seed: int = 42
    fused_cross_entropy: bool = True      # CUDA, bf16 only

    # What a token may see in a packed sequence of documents:
    #   "causal"      the whole row before it (documents run into each other)
    #   "doc_causal"  documents are separate: in-context memories read within the current
    #                 document; a Mosaics cluster memory reads the earlier documents
    attn_bias_type: str = "causal"

    # A per-head sigmoid read gate on the output of every memory / attention layer.
    memory_gate: bool = False

    def __post_init__(self) -> None:
        assert self.attn_bias_type in ATTN_BIAS_TYPES, \
            f"attn_bias_type={self.attn_bias_type!r} not in {ATTN_BIAS_TYPES}"
        assert self.init_method in INIT_METHODS, \
            f"init_method={self.init_method!r} not in {INIT_METHODS}"

    @property
    def documentwise(self) -> bool:
        return self.attn_bias_type == "doc_causal"
