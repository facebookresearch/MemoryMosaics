# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Structured metrics for training: a JSONL sink and the per-step logger.

`JsonlMetricsLogger` is the sink (one JSON object per line). `StepLogger` is what the
train loop talks to: it owns the between-log accumulators, and on a logging step it
reduces them across data parallel, writes a record, prints the console line, and
resets. Keeping the accumulators here is the point -- otherwise they leak into
`train()` as a dozen loose lists that must be appended to and cleared in lockstep.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, TextIO

import numpy as np
import torch

from training.distributed import dist_max, dist_mean, get_dp_mesh, get_global_rank

logger = logging.getLogger()

# Components shown on the console line, in this order, with their short labels. The
# JSONL record carries every component; the console line stays readable.
_CONSOLE_COMPONENTS = [("cluster", "c"), ("long", "l"), ("short", "s"),
                       ("ffn", "ffn"), ("embed", "emb")]


class JsonlMetricsLogger:
    """Writes metrics as one JSON object per line to metrics.<tag>.jsonl."""

    def __init__(self, save_dir: Path, tag: str | None = None) -> None:
        self.save_dir = save_dir
        fname = f"metrics.{tag}.jsonl" if tag else "metrics.jsonl"
        self.fpath = save_dir / fname
        self._writer: TextIO | None = None

    def open(self) -> None:
        if self._writer is None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self._writer = self.fpath.open("a")

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._writer is None:
            self.open()

        sanitized = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().item()
            sanitized[k] = v

        record = {"global_step": step, "created_at": datetime.now(UTC).isoformat()}
        record.update(sanitized)
        print(json.dumps(record), file=self._writer, flush=True)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


class StepLogger:
    """Accumulates per-step training stats and emits them every `logging_freq` steps.

    The train loop calls `record()` once per step and `maybe_log()` once per step;
    `maybe_log` decides for itself whether the step is due, so the loop has no
    logging conditional in it.

    Collective safety: `maybe_log` reduces the loss over the DP mesh, so every rank
    must call it on the same steps. It is driven by `train_state`, which advances in
    lockstep, so that holds as long as the call is unconditional in the loop.
    """

    def __init__(self, args, *, metric_logger: JsonlMetricsLogger, gpu_monitor,
                 world_mesh, dp_world_size: int, num_flop_per_token: float) -> None:
        self.args = args
        self.metric_logger = metric_logger
        self.gpu_monitor = gpu_monitor
        self.world_mesh = world_mesh
        self.dp_world_size = dp_world_size
        self.num_flop_per_token = num_flop_per_token
        self._reset()
        self.time_last_log = timer()
        # Most recent step's values, reported as-is rather than averaged.
        self.last: dict[str, Any] = {}

    def _reset(self) -> None:
        self.losses: list[float] = []
        self.data_load_times: list[float] = []
        self.fwd_times: list[float] = []
        self.bwd_times: list[float] = []
        self.mask_times: list[float] = []
        self.iter_times: list[float] = []
        self.nwords = 0

    def record(self, *, ce_loss: float, n_tokens: int,
               doc_lens: list[int], data_load_time: float, t_fwd: float,
               t_bwd: float, t_mask: float, iter_time: float) -> None:
        """Fold one step into the running window."""
        self.losses.append(ce_loss)
        self.data_load_times.append(data_load_time)
        self.fwd_times.append(t_fwd)
        self.bwd_times.append(t_bwd)
        self.mask_times.append(t_mask)
        self.iter_times.append(iter_time)
        self.nwords += n_tokens
        self.last = {
            "data_load_time": data_load_time,
            "iter_time": iter_time,
            "n_docs": len(doc_lens),
            "avg_doc_len": sum(doc_lens) / len(doc_lens),
            "max_doc_len": max(doc_lens),
            "min_doc_len": min(doc_lens),
        }

    def maybe_log(self, train_state, *, grad_norm: float, lr: float,
                  grad_components: dict[str, float],
                  update_components: dict[str, float]) -> None:
        """Emit and reset if this step is a logging step; otherwise do nothing."""
        if not train_state.should_do_at_every(self.args.logging_freq):
            return

        avg_loss, max_loss = np.mean(self.losses), np.max(self.losses)
        dp_mesh = get_dp_mesh(self.world_mesh)
        if dp_mesh is not None:
            global_avg_loss = dist_mean(avg_loss, mesh=dp_mesh)
            global_max_loss = dist_max(max_loss, mesh=dp_mesh)
        else:
            global_avg_loss, global_max_loss = float(avg_loss), float(max_loss)

        time_delta = timer() - self.time_last_log
        wps = self.nwords / time_delta
        flops = self.num_flop_per_token * wps
        gpu_mem_stats = self.gpu_monitor.get_peak_stats()

        args, last = self.args, self.last
        total_acc_steps = args.grad_acc_steps * train_state.step + train_state.acc_step
        total_tokens = self.dp_world_size * total_acc_steps * args.batch_size * args.seq_len

        def _peak(values: list[float], nd: int = 4):
            return round(max(values), nd) if values else 0

        metrics = {
            "global_step": train_state.step,
            "acc_step": train_state.acc_step,
            "loss_metrics/global_avg_loss": global_avg_loss,
            "loss_metrics/global_max_loss": global_max_loss,
            "grad_norm": grad_norm,
            **{f"grad_norm/{comp}": v for comp, v in grad_components.items()},
            **{f"delta/{comp}": v for comp, v in update_components.items()},
            "clip_cumulative": train_state.clip_cumulative,
            "wps": wps,
            "flops": flops,
            "lr": lr,
            "total_tokens": total_tokens,
            "curr_iter_time": last["iter_time"],
            "time/mask_max": _peak(self.mask_times),
            "time/fwd_max": _peak(self.fwd_times),
            "time/bwd_max": _peak(self.bwd_times),
            "time/data_max": _peak(self.data_load_times),
            "time/iter_max": _peak(self.iter_times),
            "data_load_time": last["data_load_time"],
            "docs/n_docs": last["n_docs"],
            "docs/avg_len": round(last["avg_doc_len"], 1),
            "docs/max_len": last["max_doc_len"],
            "docs/min_len": last["min_doc_len"],
            "memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
            "memory/max_active(%)": gpu_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
            "memory/num_ooms": gpu_mem_stats.num_ooms,
        }

        is_master = get_global_rank() == 0
        if is_master:
            self.metric_logger.log(metrics, step=train_state.step)
            console = self._console_line(
                train_state, global_avg_loss, grad_norm, grad_components,
                update_components, flops, wps, lr, gpu_mem_stats,
            )

        self.gpu_monitor.reset_peak_stats()
        self._reset()
        self.time_last_log = timer()

        if is_master:
            logger.info(console)

    def _console_line(self, train_state, global_avg_loss, grad_norm, grad_components,
                      update_components, flops, wps, lr, gpu_mem_stats) -> str:
        """The single human-readable line. Built before the accumulators are reset."""
        def _components(prefix: str, values: dict[str, float]) -> str:
            return "".join(f"  {prefix}_{short}: {values[name]:.2e}"
                           for name, short in _CONSOLE_COMPONENTS if name in values)

        last = self.last
        timings = ""
        if self.mask_times:
            timings = (f"  mask:{round(max(self.mask_times), 1)}"
                       f" fwd:{round(max(self.fwd_times), 1)}"
                       f" bwd:{round(max(self.bwd_times), 1)}"
                       f" data:{round(max(self.data_load_times), 1)}")
        return (
            f"step {train_state.step}"
            f"  loss: {global_avg_loss:.4f}"
            f"  grad: {grad_norm:.2e}"
            + _components("grad", grad_components)
            + _components("delta", update_components)
            + f"  flops: {flops:.2e}"
            f"  wps: {wps:.2e}"
            f"  iter: {last['iter_time']:>7}"
            f"  data: {last['data_load_time']:>7}"
            f"  lr: {lr:.2e}"
            f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
            + timings
            + f"  docs:{last['n_docs']} avg:{last['avg_doc_len']:.0f}"
            f" max:{last['max_doc_len']} min:{last['min_doc_len']}"
        )
