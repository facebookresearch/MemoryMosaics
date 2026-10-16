# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Simple checkpoint save/load using torch.distributed.checkpoint."""

import gc
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

logger = logging.getLogger()


def _disk_keys(path: Path) -> set[str]:
    from torch.distributed.checkpoint import FileSystemReader
    return set(FileSystemReader(str(path)).read_metadata().state_dict_metadata.keys())


class _ModelWrapper(Stateful):
    """The model's weights under canonical names: torch.compile's `_orig_mod.` (each
    compiled block) and DDP's `module.` are dropped, so a checkpoint loads whether or not
    the saving and loading models are compiled. Checkpoints written before names were
    canonical still load: `disk_keys` (their stored names) maps each name back."""

    def __init__(self, model: torch.nn.Module, disk_keys: set[str] | None = None) -> None:
        self.model = model
        self.on_disk: dict[str, str] = {}      # canonical name -> stored name, when they differ
        for key in disk_keys or ():
            if key.startswith("model.") and "_orig_mod." in key:
                self.on_disk[key[len("model."):].replace("_orig_mod.", "")] = key[len("model."):]

    def canonical_state_dict(self) -> dict[str, Any]:
        return get_model_state_dict(self.model)

    def state_dict(self) -> dict[str, Any]:
        return {self.on_disk.get(k, k): v for k, v in self.canonical_state_dict().items()}

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        canonical = {v: k for k, v in self.on_disk.items()}
        set_model_state_dict(self.model, {canonical.get(k, k): v for k, v in state_dict.items()},
                             options=StateDictOptions(strict=strict))


class _OptimizerWrapper(Stateful):
    """FSDP2-aware optimizer wrapper that uses the distributed state_dict API."""

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, Any]:
        return get_optimizer_state_dict(
            self.model, self.optimizer,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_optimizer_state_dict(
            self.model, self.optimizer,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    states: dict[str, Any] | None = None,
    train_config: Any | None = None,
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    state_dict: dict[str, Any] = {"model": _ModelWrapper(model)}
    if optimizer is not None:
        state_dict["optimizer"] = _OptimizerWrapper(model, optimizer)
    if states:
        state_dict.update(states)

    # Release the caching allocator's reserved-but-unused blocks before the save.
    # DCP's plan exchange goes through NCCL, and NCCL allocates OUTSIDE the PyTorch
    # caching allocator: a run whose allocator holds ~all of the card (fragmented, so
    # torch reports plenty "free" while cudaMalloc has almost nothing) makes that
    # allocation fail as "NCCL Error 1: unhandled cuda error" -- and since the directory
    # is created on entry, the failure leaves an empty step_N/ behind. Costs one
    # cudaFree sync per checkpoint, which is negligible at checkpoint_freq.
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    dcp.save(state_dict, checkpoint_id=str(path))

    # Save per-rank extra state (e.g. ClusterDataIterator pipeline states) that
    # DCP cannot handle because DCP only saves non-tensor values from rank 0.
    if states:
        for v in states.values():
            if hasattr(v, "save_extra"):
                v.save_extra(path)

    if train_config is not None and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        import dataclasses
        config_dict = dataclasses.asdict(train_config)
        with open(path / "params.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    logger.info(f"Checkpoint saved to {path}")


def _load_model_lenient(path: Path, model: torch.nn.Module) -> None:
    """Load model weights, silently skipping parameters absent from the checkpoint.

    Useful for visualization tools when the live model has parameters that were
    added after an older checkpoint was saved. Missing parameters keep their
    init_weights() values; extra keys in the checkpoint are ignored.
    """
    on_disk = _disk_keys(path)
    wrapper = _ModelWrapper(model, on_disk)
    live = wrapper.state_dict()                       # stored names -> live tensors
    to_load = {k: v.clone() for k, v in live.items() if "model." + k in on_disk}
    if to_load:
        dcp.load({"model": to_load}, checkpoint_id=str(path))
        wrapper.load_state_dict(to_load, strict=False)

    missing = [k for k in live if "model." + k not in on_disk]
    extra = [k for k in on_disk if k.startswith("model.") and k[len("model."):] not in live]
    if missing:
        logger.info(f"Checkpoint missing {len(missing)} key(s), kept initialized: {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if extra:
        logger.info(f"Checkpoint has {len(extra)} extra key(s), ignored: {extra[:5]}{'…' if len(extra) > 5 else ''}")


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    states: dict[str, Any] | None = None,
    strict: bool = True,
) -> None:
    path = Path(path)
    assert path.exists(), f"Checkpoint not found: {path}"
    assert is_complete_checkpoint(path), (
        f"Incomplete checkpoint {path}: no .metadata, so the save was interrupted "
        f"(the directory is created before anything is written). Delete it and resume "
        f"from an earlier step."
    )

    if not strict:
        _load_model_lenient(path, model)
        logger.info(f"Checkpoint loaded (lenient) from {path}")
        return

    state_dict: dict[str, Any] = {"model": _ModelWrapper(model, _disk_keys(path))}
    if optimizer is not None:
        state_dict["optimizer"] = _OptimizerWrapper(model, optimizer)
    if states:
        state_dict.update(states)

    # Workaround for pytorch/pytorch#138575: dcp.load() replaces Stateful values
    # in the dict with new objects, breaking future saves. Save and restore the
    # original wrappers so the dict keeps pointing at the live objects.
    original_stateful = {k: v for k, v in state_dict.items() if isinstance(v, Stateful)}
    dcp.load(state_dict, checkpoint_id=str(path))
    state_dict.update(original_stateful)

    # Load per-rank extra state (e.g. ClusterDataIterator pipeline states) that
    # was saved outside DCP. This must come after dcp.load so that load_state_dict
    # has already been called and self._step is set.
    if states:
        for v in states.values():
            if hasattr(v, "load_extra"):
                v.load_extra(path)

    logger.info(f"Checkpoint loaded from {path}")


def print_checkpoint_keys(path: str | Path) -> None:
    """Print all tensor keys stored in a distributed checkpoint."""
    from torch.distributed.checkpoint import FileSystemReader
    metadata = FileSystemReader(str(path)).read_metadata()
    keys = sorted(metadata.state_dict_metadata.keys())
    print(f"Checkpoint at {path} — {len(keys)} keys:")
    for k in keys:
        print(f"  {k}")


def is_complete_checkpoint(path: str | Path) -> bool:
    """Whether a step_* directory holds a loadable checkpoint.

    DCP writes each rank's shard first and the coordinator's `.metadata` LAST, so
    `.metadata` is present only once the save finished. Its absence means the save was
    interrupted -- and save_checkpoint mkdir's the directory on entry, before writing
    anything, so a job killed during dcp.save (wall-clock, OOM, NCCL) leaves behind
    exactly an empty step_N/.
    """
    return (Path(path) / ".metadata").is_file()


def get_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Newest COMPLETE step_* checkpoint, or None.

    Incomplete directories are skipped rather than returned. Selecting purely by the
    highest step number meant an empty step_N/ passed load_checkpoint's `path.exists()`
    assert and then failed deep inside dcp.load, so a single interrupted save made every
    subsequent resume fail instead of falling back to the previous good checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    steps = []
    for d in checkpoint_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                steps.append(int(d.name.split("_")[1]))
            except ValueError:
                continue

    for step in sorted(steps, reverse=True):
        path = checkpoint_dir / f"step_{step}"
        if is_complete_checkpoint(path):
            return path
        logger.warning(
            f"Skipping incomplete checkpoint {path} (no .metadata; the save was "
            f"interrupted). Delete it to stop this warning."
        )
    return None
