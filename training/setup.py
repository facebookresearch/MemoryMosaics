# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Run setup: config resolution, distributed init, model construction, resume.

These are the phases `train()` runs before entering the loop, in this order:

    resolve_config -> setup_distributed -> build_model -> resume_if_available

They are separate functions rather than one `setup_training()` returning a dozen
values, so the data flow between phases stays visible at the call site.

Ordering is load-bearing in two places. `resolve_config` must run before
`setup_distributed`, because the resolved-hyperparameter dump is only meaningful
once vocab_size / max_seq_len / checkpoint.path are filled in (and dp_size, which
`setup_distributed` fills in just before logging). `build_model` must run after the
mesh exists, because parallelize_model shards against it.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed

from model import build_model as build_family_model
from training.checkpoint import get_latest_checkpoint, load_checkpoint
from training.config import TrainConfig
from training.distributed import (
    TORCH_DTYPE_MAP,
    ParallelDims,
    get_dp_rank,
    get_dp_world_size,
    get_global_rank,
    get_world_size,
    init_distributed,
    parallelize_model,
)

logger = logging.getLogger()


def _flatten_config(d: dict, prefix: str = "") -> list:
    """Flatten a nested asdict() config into [(dotted.key, value), ...] in field order."""
    out = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.extend(_flatten_config(v, key + "."))
        else:
            out.append((key, v))
    return out


def resolve_config(args: TrainConfig, tokenizer) -> tuple[Path, int | None]:
    """Fill in config values derived from the tokenizer/paths and prepare the dump dir.

    Mutates `args` in place: vocab_size, max_seq_len and checkpoint.path are derived
    here rather than required in the YAML. Returns (dump_path, doc_sep_id).
    """
    if args.model.vocab_size < 0:
        args.model.vocab_size = tokenizer.n_words
    assert args.model.vocab_size == tokenizer.n_words

    doc_sep_id: int | None = args.data.doc_sep_id
    if doc_sep_id is not None:
        logger.info(f"doc_sep_id={doc_sep_id} ({tokenizer.decode([doc_sep_id])!r})")

    assert args.dump_dir
    dump_path = Path(args.dump_dir)

    if args.checkpoint.path is None:
        args.checkpoint.path = str(dump_path / "checkpoints")

    args.model.max_seq_len = args.seq_len

    if get_global_rank() == 0:
        dump_path.mkdir(parents=True, exist_ok=True)
        # Re-arm the SLURM requeue guard: a fresh launch has not completed yet.
        # Clearing a stale marker means a run relaunched with a raised step count
        # is not falsely treated as finished (see the completion marker train()
        # writes at the end, and launchers/slurm.py).
        (dump_path / "training_complete").unlink(missing_ok=True)

    return dump_path, doc_sep_id


def setup_distributed(args: TrainConfig):
    """Initialise the process group and build the device mesh.

    Returns (world_mesh, parallel_dims, dp_rank, dp_world_size).
    """
    init_distributed()

    world_size = get_world_size()
    args.distributed.dp_size = world_size

    if get_global_rank() == 0:
        # Log the fully-resolved hyperparameters (after vocab_size / max_seq_len / dp_size /
        # checkpoint.path are filled in) so the ACTUAL values in effect -- not just what the
        # YAML set -- are visible. Both a flat "section.key = value" listing (easy to scan /
        # grep) and the JSON block (same content as the saved params.json).
        cfg = asdict(args)
        flat = "\n".join(f"  {k} = {v}" for k, v in _flatten_config(cfg))
        logger.info("Resolved hyperparameters:\n" + flat)
        logger.info("Resolved hyperparameters (JSON):\n" + json.dumps(cfg, indent=2, default=str))

    parallel_dims = ParallelDims(dp=world_size, world_size=world_size,
                                 dp_type=args.distributed.dp_type)
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    dp_world_size = get_dp_world_size(world_mesh)
    dp_rank = get_dp_rank(world_mesh)

    logger.info(f"dp_rank={dp_rank}, dp_world_size={dp_world_size}")
    return world_mesh, parallel_dims, dp_rank, dp_world_size


def get_num_flop_per_token(num_non_embed_params: int, n_layers: int, dim: int,
                           seq_length: int) -> float:
    """Training FLOPs per token: 6 per parameter plus an attention-score term."""
    attn_flops = 3.5 * (4 * n_layers * seq_length * dim // 2)
    return 6 * num_non_embed_params + attn_flops


def build_model(args: TrainConfig, world_mesh, parallel_dims) -> tuple[torch.nn.Module, float]:
    """Construct, shard and materialise the model. Returns (model, num_flop_per_token).

    Built on the meta device first so no rank ever allocates the unsharded parameters:
    `parallelize_model` decides the sharding, and only then does `to_empty` allocate
    each rank's actual slice on the GPU.
    """
    torch.manual_seed(args.seed)

    logger.info(f"Building {args.model_type} model on meta device")
    with torch.device("meta"):
        model = build_family_model(args.model_type, args.model)

    parallelize_model(
        model, world_mesh, parallel_dims,
        param_dtype=TORCH_DTYPE_MAP[args.distributed.model_dtype],
        compile=args.distributed.compile,
        ac_mode=args.distributed.ac_mode,
        selective_ac_option=args.distributed.selective_ac_option,
    )
    model = model.to_empty(device="cuda")
    model.init_weights()
    model.train()

    num_params = sum(p.numel() for p in model.parameters())
    num_non_embed = num_params - sum(p.numel() for p in model.tok_embeddings.parameters())
    num_flop_per_token = get_num_flop_per_token(
        num_non_embed, args.model.n_layers, args.model.dim, args.seq_len
    )
    logger.info(f"Model size: {num_params:,} parameters")
    return model, num_flop_per_token


def resume_if_available(args: TrainConfig, ckpt_dir: Path, *, model, optimizer,
                        scheduler, data_iterator, train_state) -> None:
    """Restore from the latest checkpoint in `ckpt_dir`, else optionally warm-start.

    A checkpoint in `ckpt_dir` is a resume: model, optimizer, scheduler, step count
    and data-loader position all come back, and `continue_from` is ignored. With no
    checkpoint, `continue_from` loads weights only -- a fresh run off trained
    parameters, starting from step 0.
    """
    latest = get_latest_checkpoint(ckpt_dir)
    if latest is not None:
        # Fork the data-loader producer BEFORE load_checkpoint so it is forked
        # before dcp.load triggers NCCL initialisation.  Forking after NCCL
        # creates background threads whose inherited mutexes can deadlock the child.
        if hasattr(data_iterator, "prepare_resume"):
            data_iterator.prepare_resume(latest)
        load_checkpoint(latest, model, optimizer, {
            "train_state": train_state,
            "lr_scheduler": scheduler,
            "data_iterator": data_iterator,
        })
        logger.info(f"Resumed from {latest}, step={train_state.step}")
    elif args.continue_from is not None and args.continue_from.checkpoint_dir:
        load_checkpoint(args.continue_from.checkpoint_dir, model)
        logger.info(f"Loaded weights from {args.continue_from.checkpoint_dir}")
