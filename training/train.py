# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Training loop for Memory Mosaics."""

import gc
import logging
import os
import shutil
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Any

import torch
import torch.distributed
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.stateful import Stateful

from model.common.mask import tokens_to_seq_lens
from tokenizer.tokenizer import Tokenizer
from training.checkpoint import save_checkpoint
from training.config import TrainConfig
from dataloader import build_data_iterator
from training.distributed import get_global_rank
from training.gpu_monitor import GPUMemoryMonitor
from training.metrics import JsonlMetricsLogger, StepLogger
from training.optimizer import build_optimizer
from training.probes import (
    compute_component_grad_norms,
    compute_component_update_norms,
)
from training.setup import (
    build_model,
    resolve_config,
    resume_if_available,
    setup_distributed,
)

logger = logging.getLogger()

# Unset rather than "": an empty TORCH_LOGS breaks `import torch` on some builds, e.g. in
# the data loader's process when it is spawned rather than forked.
os.environ.pop("TORCH_LOGS", None)
os.environ["TORCHDYNAMO_VERBOSE"] = "0"
os.environ.pop("TORCH_COMPILE_DEBUG", None)
os.environ.pop("TORCHINDUCTOR_TRACE", None)

torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.recompile_limit = 1000
torch._dynamo.config.verbose = False
# The model calls opaque custom ops (torch.library.custom_op: the leaky-average recurrence
# in model/mosaics/leaky_average.py, the fused cross-entropy in model/common/cross_entropy.py). Under
# `ac_mode: full` the block is wrapped in a non-reentrant activation-checkpoint HOP and
# then compiled; dynamo's checkpoint-HOP tracer conservatively treats any opaque custom op
# as a possible external side effect and refuses ("HOP: Unsafe side effect ... Mutating a variable from
# outside the scope"). Our checkpointed forwards are all side-effect-free (these ops are
# pure value producers), so allowing the trace is safe: the op is a normal graph node that
# recomputes correctly in the AC backward, and there is no real forward mutation to skip.
# hasattr-guarded: some torch builds lack this knob and ConfigModule raises on unknown keys.
if hasattr(torch._dynamo.config, "skip_fwd_side_effects_in_bwd_under_checkpoint"):
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True

import logging as _logging
for _name in [
    "torch._dynamo", "torch._inductor", "torch._functorch",
    "torch._dynamo.output_graph", "torch._dynamo.guards",
    "torch.fx.experimental.symbolic_shapes",
]:
    _logging.getLogger(_name).setLevel(_logging.ERROR)


@dataclass
class TrainState(Stateful):
    step: int
    acc_step: int
    clip_cumulative: int

    def update_steps(self, grad_acc_steps: int) -> None:
        self.acc_step += 1
        self.acc_step = self.acc_step % grad_acc_steps
        if self.acc_step == 0:
            self.step += 1

    def should_do_optim(self) -> bool:
        return self.acc_step == 0

    def should_do_at_every(self, step_interval: int) -> bool:
        return self.acc_step == 0 and (self.step % step_interval) == 0

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "acc_step": self.acc_step, "clip_cumulative": self.clip_cumulative}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.clip_cumulative = state_dict["clip_cumulative"]


_preemption_flag = {"flag": False}


def _set_preemption_flag(signum, frame):
    logger.warning(f"Signal {signum} received — exiting at the end of this step without checkpointing "
                   "(resumes from the last periodic checkpoint).")
    _preemption_flag["flag"] = True


def _inherit_model(raw: dict, config_path: str) -> dict:
    """continue_from.inherit_model: model_type, the model settings and the tokenizer of the
    checkpoint's run (its params.json), with the config's own `model:` keys on top."""
    import json

    ckpt = raw["continue_from"].get("checkpoint_dir")
    if not ckpt:
        raise ValueError(f"{config_path}: continue_from.inherit_model needs continue_from.checkpoint_dir")
    params_path = Path(ckpt) / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"{config_path}: continue_from.inherit_model reads {params_path}, "
                                f"which does not exist")
    with open(params_path) as f:
        params = json.load(f)
    if raw.get("model_type", params["model_type"]) != params["model_type"]:
        raise ValueError(f"{config_path}: model_type {raw['model_type']!r}, but the checkpoint "
                         f"is a {params['model_type']!r} model")
    data = dict(raw.get("data") or {})
    data.setdefault("tokenizer_path", (params.get("data") or {}).get("tokenizer_path"))
    inherited = dict(params["model"])
    return {**raw, "model_type": params["model_type"],
            "model": {**inherited, **(raw.get("model") or {})}, "data": data}


def _load_config(config_path: str) -> TrainConfig:
    import typing
    import yaml
    from dataclasses import fields

    from model import FAMILIES

    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if (raw.get("continue_from") or {}).get("inherit_model"):
        raw = _inherit_model(raw, config_path)
    model_type = raw.get("model_type", "mosaics")
    if model_type not in FAMILIES:
        raise ValueError(f"{config_path}: model_type {model_type!r} not in {sorted(FAMILIES)}")

    def _coerce(val, fld_type):
        """Coerce a parsed YAML value to the declared field type."""
        origin = getattr(fld_type, "__origin__", None)
        if origin is typing.Union or origin is type(int | None):
            args = [a for a in typing.get_args(fld_type) if a is not type(None)]
            if args:
                return _coerce(val, args[0])
        if fld_type is float and isinstance(val, str):
            return float(val)
        if fld_type is int and isinstance(val, str):
            return int(val)
        if fld_type is bool and isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        if fld_type is float and isinstance(val, int):
            return float(val)
        return val

    def _fill_dataclass(cls, data, where="config"):
        if data is None:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"{config_path}: unknown key(s) {unknown} in {where} "
                             f"({cls.__name__}); valid keys: {sorted(known)}")
        kwargs = {}
        for fld in fields(cls):
            if fld.name in data:
                val = data[fld.name]
                fld_type = fld.type
                if cls is TrainConfig and fld.name == "model":     # the family's config
                    fld_type = FAMILIES[model_type][0]
                if hasattr(fld_type, "__dataclass_fields__"):
                    val = _fill_dataclass(fld_type, val, f"{where}.{fld.name}")
                elif isinstance(val, dict) and not hasattr(fld.type, "__dataclass_fields__"):
                    fld_origin = getattr(fld.type, "__origin__", None)
                    if fld_origin is not dict:
                        for sub_type in typing.get_args(fld.type):
                            if hasattr(sub_type, "__dataclass_fields__"):
                                val = _fill_dataclass(sub_type, val, f"{where}.{fld.name}")
                                break
                else:
                    val = _coerce(val, fld.type)
                kwargs[fld.name] = val
        return cls(**kwargs)

    raw = dict(raw)
    raw.setdefault("model", {})            # the family's defaults, not MosaicsConfig's
    return _fill_dataclass(TrainConfig, raw)


def _prune_checkpoints(ckpt_dir: Path, keep_latest: int) -> None:
    """Delete oldest step_* checkpoints, keeping only the `keep_latest` most recent."""
    if keep_latest <= 0 or get_global_rank() != 0:
        return
    dirs = sorted(
        (d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    for old in dirs[:-keep_latest]:
        shutil.rmtree(old)
        logger.info(f"Pruned old checkpoint: {old}")


def train(args: TrainConfig) -> None:
    tokenizer = Tokenizer(args.data.tokenizer_path)
    dump_path, doc_sep_id = resolve_config(args, tokenizer)

    signal.signal(signal.SIGUSR2, _set_preemption_flag)
    world_mesh, parallel_dims, dp_rank, dp_world_size = setup_distributed(args)
    model, num_flop_per_token = build_model(args, world_mesh, parallel_dims)

    gpu_monitor = GPUMemoryMonitor("cuda")
    logger.info(
        f"GPU capacity: {gpu_monitor.device_name} ({gpu_monitor.device_index}) "
        f"with {gpu_monitor.device_capacity_gib:.2f}GiB memory"
    )
    logger.info(f"GPU memory usage: {gpu_monitor}")

    metric_logger = JsonlMetricsLogger(dump_path, tag="train")
    metric_logger.open()

    optimizer, scheduler = build_optimizer(model, args.optim, args.steps)
    data_iterator = build_data_iterator(args, tokenizer, dp_rank=dp_rank, dp_world_size=dp_world_size)
    batch_loader = iter(data_iterator)

    train_state = TrainState(step=0, acc_step=0, clip_cumulative=0)
    ckpt_dir = Path(args.checkpoint.path)
    resume_if_available(args, ckpt_dir, model=model, optimizer=optimizer,
                        scheduler=scheduler, data_iterator=data_iterator,
                        train_state=train_state)

    step_logger = StepLogger(
        args,
        metric_logger=metric_logger,
        gpu_monitor=gpu_monitor,
        world_mesh=world_mesh,
        dp_world_size=dp_world_size,
        num_flop_per_token=num_flop_per_token,
    )
    saved = False
    component_grad_norms: dict[str, float] = {}
    component_update_norms: dict[str, float] = {}

    while train_state.step < args.steps:
        train_state.update_steps(args.grad_acc_steps)

        if train_state.should_do_at_every(args.gc_collect_freq):
            gc.collect()

        # ── Data loading (timed) ──
        curr_lr = float(optimizer.param_groups[0]["lr"])
        data_load_start = timer()
        batch = next(batch_loader)

        input_ids = torch.from_numpy(batch.x).cuda()
        labels = torch.from_numpy(batch.y).cuda()
        labels_mask = None if batch.mask is None else torch.from_numpy(batch.mask).cuda()

        data_load_time = round(timer() - data_load_start, 4)

        # ── Forward / backward (timed with CUDA events) ──
        start_timer = torch.cuda.Event(enable_timing=True)
        end_timer = torch.cuda.Event(enable_timing=True)
        start_timer.record()

        t_mask_start = timer()
        seq_lens = tokens_to_seq_lens(input_ids, tokenizer.eos_id)

        if doc_sep_id is not None:
            input_ids[input_ids == tokenizer.eos_id] = doc_sep_id

        attn_mask = model.create_attention_mask(input_ids, seq_lens, step=train_state.step, seed=args.seed)
        torch.cuda.synchronize()
        t_mask = round(timer() - t_mask_start, 4)

        t_fwd_start = timer()
        ce_loss = model(input_ids, mask=attn_mask, targets=labels, targets_mask=labels_mask)
        torch.cuda.synchronize()
        t_fwd = round(timer() - t_fwd_start, 4)

        t_bwd_start = timer()
        loss = ce_loss / args.grad_acc_steps
        loss.backward()
        torch.cuda.synchronize()
        t_bwd = round(timer() - t_bwd_start, 4)

        ce_loss = ce_loss.detach()

        if train_state.should_do_optim():
            # Per-component grad norms (raw, pre-clip) + relative update
            # ‖Δw‖/‖w‖ — only on logging steps (snapshot params pre-step).
            _is_log_step = train_state.should_do_at_every(args.logging_freq)
            _prev_params = None
            if _is_log_step:
                component_grad_norms = compute_component_grad_norms(model)
                _prev_params = {
                    n: p.detach().clone()
                    for n, p in model.named_parameters() if p.requires_grad
                }
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.optim.clip)
            grad_norm = grad_norm.full_tensor().item() if isinstance(grad_norm, DTensor) else grad_norm.item()
            if grad_norm > args.optim.clip:
                train_state.clip_cumulative += 1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # Update norms need the pre-step snapshot, so they are only computed on
            # steps that will actually log.
            if _is_log_step:
                component_update_norms = compute_component_update_norms(model, _prev_params)
                del _prev_params
        else:
            grad_norm = 0.0

        end_timer.record()
        torch.cuda.synchronize()
        curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

        step_logger.record(
            ce_loss=ce_loss.item(),
            n_tokens=input_ids.numel(),
            doc_lens=seq_lens,
            data_load_time=data_load_time,
            t_fwd=t_fwd,
            t_bwd=t_bwd,
            t_mask=t_mask,
            iter_time=curr_iter_time,
        )

        # ── Logging (StepLogger decides whether this step is due) ──
        step_logger.maybe_log(
            train_state,
            grad_norm=grad_norm,
            lr=curr_lr,
            grad_components=component_grad_norms,
            update_components=component_update_norms,
        )

        if args.checkpoint_freq > 0 and train_state.should_do_at_every(args.checkpoint_freq):
            save_path = ckpt_dir / f"step_{train_state.step}"
            save_checkpoint(
                save_path, model, optimizer,
                {"train_state": train_state, "lr_scheduler": scheduler, "data_iterator": data_iterator},
                train_config=args,
            )
            _prune_checkpoints(ckpt_dir, args.checkpoint.keep_latest)
            saved = True

        if _preemption_flag["flag"]:
            # Exit without checkpointing: SLURM requeues the job and it resumes from
            # the last periodic checkpoint, losing at most `checkpoint_freq` steps.
            # Saving here would race the preemption grace period and can leave a
            # half-written checkpoint that the resume path then refuses to load.
            sys.exit(0)

    if not saved:
        save_path = ckpt_dir / f"step_{train_state.step}"
        save_checkpoint(
            save_path, model, optimizer,
            {"train_state": train_state, "lr_scheduler": scheduler, "data_iterator": data_iterator},
            train_config=args,
        )
        _prune_checkpoints(ckpt_dir, args.checkpoint.keep_latest)

    # Reaching here means the loop ran to args.steps (the preemption path
    # sys.exit(0)s inside the loop, so it never gets here). Drop a marker the
    # SLURM launcher checks so a finished run is NOT requeued if it happens to
    # exit non-zero during distributed teardown.
    if get_global_rank() == 0:
        (dump_path / "training_complete").touch()

    metric_logger.close()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = None
    for arg in sys.argv[1:]:
        if arg.startswith("config="):
            config_path = arg.split("=", 1)[1]
    assert config_path is not None, "Usage: python -m training.train config=path/to/config.yaml"
    args = _load_config(config_path)
    if args.distributed.ac_mode != "none":
        os.environ["precompile_flex_attention"] = "true"
    train(args)


if __name__ == "__main__":
    main()
