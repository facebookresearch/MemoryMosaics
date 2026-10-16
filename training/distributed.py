# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Distributed training: data parallelism (FSDP2 or DDP), activation checkpointing,
and per-block torch.compile. Launch with torchrun or srun (see launchers/slurm.py)."""

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

import torch
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed._composable.replicate import replicate
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh

from model.common.lm import LanguageModel

logger = logging.getLogger()

TORCH_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass
class ParallelDims:
    dp: int
    world_size: int
    dp_type: str = "fsdp"

    def __post_init__(self) -> None:
        self.dp_type = self.dp_type.lower()
        assert self.dp == self.world_size >= 1, f"dp({self.dp}) != world_size({self.world_size})"
        assert self.dp_type in ("fsdp", "ddp")

    def build_mesh(self, device_type: str = "cuda") -> DeviceMesh:
        dims, names = ((self.dp,), ("dp",)) if self.dp > 1 else ((), ())
        logger.info(f"Building {len(dims)}-D device mesh: {list(names)} = {list(dims)}")
        return init_device_mesh(device_type, mesh_shape=dims, mesh_dim_names=names)

    @property
    def dp_enabled(self) -> bool:
        return self.dp > 1


def _is_torchrun() -> bool:
    return "LOCAL_RANK" in os.environ


def _is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ and not _is_torchrun()


def _get_slurm_master_addr() -> str:
    import subprocess
    hostnames = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
    )
    return hostnames.split()[0].decode("utf-8")


def _get_slurm_master_port() -> int:
    import random as _random
    rng = _random.Random(int(os.environ.get("SLURM_JOB_ID", -1)))
    return rng.randint(20000, 60000)


def init_distributed(timeout: int = 1800) -> None:
    """Initialize the process group. `timeout` (seconds) bounds collective ops
    such as the GATHER inside dcp.save; raised to 1800s to tolerate slow shared
    filesystems during distributed checkpointing."""
    if _is_torchrun():
        local_rank = int(os.environ["LOCAL_RANK"])
    elif _is_slurm_job():
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        local_rank = 0

    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(local_rank)

    if torch.distributed.is_initialized():
        return

    if _is_torchrun():
        torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=timeout))
    elif _is_slurm_job():
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        master_addr = _get_slurm_master_addr()
        master_port = _get_slurm_master_port()
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://{master_addr}:{master_port}",
            timeout=timedelta(seconds=timeout),
        )


def get_world_size() -> int:
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def get_global_rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def get_dp_mesh(world_mesh: DeviceMesh) -> DeviceMesh | None:
    if world_mesh.ndim == 0:
        return None
    if "dp" not in world_mesh.mesh_dim_names:
        return None
    return world_mesh["dp"]


def get_dp_rank(world_mesh: DeviceMesh) -> int:
    dp_mesh = get_dp_mesh(world_mesh)
    return dp_mesh.get_local_rank() if dp_mesh is not None else 0


def get_dp_world_size(world_mesh: DeviceMesh) -> int:
    dp_mesh = get_dp_mesh(world_mesh)
    return dp_mesh.size() if dp_mesh is not None else 1


def parallelize_model(
    model: LanguageModel,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype = torch.float32,
    compile: bool = True,
    ac_mode: str = "none",
    selective_ac_option: str | int = 2,
) -> None:
    """Apply activation checkpointing, data parallelism, then per-block torch.compile.

    The model should be on the meta device before calling this.
    """
    if ac_mode != "none":
        _apply_ac(model, ac_mode, selective_ac_option)

    if parallel_dims.dp_enabled:
        if parallel_dims.dp_type == "fsdp":
            _apply_fsdp(model, world_mesh, param_dtype=param_dtype, reduce_dtype=reduce_dtype)
        else:
            replicate(model, device_mesh=world_mesh, bucket_cap_mb=100)
            logger.info("Applied DDP to the model")
    else:
        for param in model.parameters():
            param.data = param.data.to(dtype=param_dtype)

    if compile:
        _apply_blockwise_compile(model)


def _apply_blockwise_compile(model: LanguageModel, fullgraph: bool = True) -> None:
    """Compile each block individually.

    fullgraph=True forbids graph breaks. A model containing a DeltaNet (the rnn family,
    Hybrid DeltaNet, or a deltanet short memory) compiles with fullgraph=False whatever
    the caller asks, since the fla kernel is excluded from compilation. Detected by
    walking the model rather than from config flags.
    """
    from model.common.deltanet import DeltaNet

    n_deltanet = sum(isinstance(m, DeltaNet) for m in model.modules())
    if n_deltanet:
        fullgraph = False
        logger.info(f"Found {n_deltanet} DeltaNet module(s) -> forcing fullgraph=False")
    logger.info(f"Compiling each MosaicBlock with torch.compile (fullgraph={fullgraph})")
    for layer_id, block in model.layers.named_children():
        compiled = torch.compile(block, fullgraph=fullgraph)
        model.layers.register_module(layer_id, compiled)
    if model.loss_fn is not None:
        model.loss_fn = torch.compile(model.loss_fn)


def _apply_fsdp(
    model: torch.nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
) -> None:
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    for layer_id, block in enumerate(model.layers):
        reshard_after_forward = layer_id < len(model.layers) - 1
        fully_shard(block, **fsdp_config, reshard_after_forward=reshard_after_forward)
    fully_shard(model, **fsdp_config, reshard_after_forward=True)
    logger.info("Applied FSDP to the model")


_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
}


def _apply_ac(model: torch.nn.Module, ac_mode: str, selective_ac_option: str | int) -> None:
    for layer_id, block in model.layers.named_children():
        if ac_mode == "full":
            block = ptd_checkpoint_wrapper(block, preserve_rng_state=False)
        elif ac_mode == "selective":
            if selective_ac_option == "op":
                from torch.utils.checkpoint import (
                    CheckpointPolicy,
                    create_selective_checkpoint_contexts,
                )

                def _get_custom_policy(meta):
                    def _custom_policy(ctx, func, *args, **kwargs):
                        mode = "recompute" if ctx.is_recompute else "forward"
                        mm_count_key = f"{mode}_mm_count"
                        if func == torch.ops.aten.mm.default:
                            meta[mm_count_key] += 1
                        to_save = func in _save_list and not (
                            func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                        )
                        return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE
                    return _custom_policy

                def selective_context_fn():
                    return create_selective_checkpoint_contexts(_get_custom_policy(defaultdict(int)))

                block = ptd_checkpoint_wrapper(block, context_fn=selective_context_fn, preserve_rng_state=False)
            elif isinstance(selective_ac_option, int):
                ac_freq = int(selective_ac_option)
                ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
                ptd_checkpoint_wrapper._count += 1
                if not ac_freq or ptd_checkpoint_wrapper._count % ac_freq == 0:
                    block = ptd_checkpoint_wrapper(block, preserve_rng_state=False)
        model.layers.register_module(layer_id, block)
    logger.info(f"Applied {ac_mode} activation checkpointing")


def dist_mean(value: float, mesh: DeviceMesh | None = None) -> float:
    t = torch.tensor(value, device="cuda")
    if mesh is not None:
        torch.distributed.all_reduce(t, group=mesh.get_group())
        t /= mesh.size()
    return t.item()


def dist_max(value: float, mesh: DeviceMesh | None = None) -> float:
    t = torch.tensor(value, device="cuda")
    if mesh is not None:
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=mesh.get_group())
    return t.item()
