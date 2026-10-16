# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Configuration dataclasses for training Memory Mosaics."""

from dataclasses import dataclass, field

from model import FAMILIES
from model.common.config import BaseConfig
from model.mosaics.config import MosaicsConfig


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.95
    fused_optimizer: bool = True
    clip: float = 1.0

    scheduler: str = "cosine"
    warmup: int = 2000
    lr_min_ratio: float = 0.1
    cycle_length: float = 1.0
    cosine_theta: float = 1.0
    exp_factor: float = 0.5


@dataclass
class DataConfig:
    loader: str = "pretrain"  # "pretrain" | "cluster" | "sft"
    seed: int = 42
    # Wrap each pretraining document in [BOS] ... [EOS]. EOS is the document boundary
    # that document-wise attention (attn_bias_type: doc_causal) splits on.
    add_bos: bool = True
    add_eos: bool = True
    # Token ID to replace EOS with in input_ids after document-boundary detection.
    # None = no replacement. Use -1 as a sentinel for "unset" in YAML if needed.
    doc_sep_id: int | None = None
    prefetch_size: int = 16
    shuffle_buffer_size: int = 64

    # HuggingFace repo id or local directory of the tokenizer (None: Mistral-7B's).
    tokenizer_path: str | None = None

    # Pretrain source directories with weights
    sources: dict[str, float] = field(default_factory=dict)
    root_dir: str | None = None

    # Cluster loader fields
    cluster_data_dir: str | None = None
    cluster_metadata: str = "metadata.json"  # use metadata_train.json for train-only clusters
    max_cached_clusters: int = 256
    # Cluster draws ("slots") per step, shared evenly by the batch_size x world-size sequences;
    # 0 (or more than the sequences) = one slot per sequence (max diversity).
    clusters_per_step: int = 32
    # {n_clusters: n_slots} — e.g. {1: 28, 2: 3, 3: 1}: 28 slots of 1 cluster, 3 slots mixing
    # 2 clusters, 1 slot mixing 3 (sum must equal clusters_per_step).
    # None → {1: clusters_per_step}: every sequence from a single cluster.
    cluster_frac: dict | None = None

    # SFT: a directory of JSONL conversations ({"messages": [...]}), see tokenizer.chat_template.
    sft_data_path: str | None = None

    def __post_init__(self) -> None:
        assert self.loader in ("pretrain", "cluster", "sft"), f"Unknown loader: {self.loader}"


@dataclass
class DistributedConfig:
    dp_size: int = 1                      # set at launch: the number of processes
    dp_type: str = "fsdp"                 # "fsdp" | "ddp"
    model_dtype: str = "bf16"
    compile: bool = True
    ac_mode: str = "none"  # "none" | "full" | "selective"
    selective_ac_option: str | int = 2


@dataclass
class CheckpointConfig:
    path: str | None = None
    keep_latest: int = -1


@dataclass
class ContinueFromConfig:
    """Start a NEW run from another run's weights (optimizer and data state are not
    restored), e.g. fine-tuning a pretrained model. Ignored once the run has checkpoints
    of its own: it then resumes.

    inherit_model: take model_type, the model settings and the tokenizer from the
    checkpoint's params.json, so one fine-tuning config serves any pretrained model; keys
    under `model:` in the config override the inherited ones."""
    checkpoint_dir: str = ""
    inherit_model: bool = False


@dataclass
class TrainConfig:
    name: str = "memory_mosaics"
    model_type: str = "mosaics"  # "mosaics" | "transformer" | "rnn": the model family
    dump_dir: str = ""
    seed: int = 42
    batch_size: int = 2
    grad_acc_steps: int = 1
    seq_len: int = 2048

    checkpoint_freq: int = 100
    gc_collect_freq: int = 1000
    logging_freq: int = 10

    steps: int = 1000

    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    # The family's config (MosaicsConfig, TransformerConfig or RNNConfig, by model_type).
    model: BaseConfig = field(default_factory=MosaicsConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    continue_from: ContinueFromConfig | None = None

    def __post_init__(self) -> None:
        assert self.model_type in FAMILIES, f"model_type={self.model_type!r} not in {sorted(FAMILIES)}"
        expected = FAMILIES[self.model_type][0]
        assert isinstance(self.model, expected), \
            f"model_type {self.model_type!r} needs a {expected.__name__}, got {type(self.model).__name__}"
        assert self.data.add_eos or not self.model.documentwise, \
            "attn_bias_type doc_causal splits documents at EOS: data.add_eos must be true"
        assert self.distributed.model_dtype == "bf16" or not self.model.fused_cross_entropy, \
            f"the fused cross-entropy kernel is bf16-only: set model.fused_cross_entropy: false " \
            f"for model_dtype {self.distributed.model_dtype}"
