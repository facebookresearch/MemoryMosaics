# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""GPU memory monitoring."""

import logging
from typing import NamedTuple

import torch

logger = logging.getLogger()


class GPUMemStats(NamedTuple):
    max_active_gib: float
    max_active_pct: float
    max_reserved_gib: float
    max_reserved_pct: float
    num_alloc_retries: float
    num_ooms: float


class GPUMemoryMonitor:
    def __init__(self, device: str = "cuda:0") -> None:
        self.device = torch.device(device)
        self.device_name = torch.cuda.get_device_name(self.device)
        self.device_index = torch.cuda.current_device()
        self.device_capacity = torch.cuda.get_device_properties(self.device).total_memory
        self.device_capacity_gib = self.device_capacity / (1024**3)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    def get_peak_stats(self) -> GPUMemStats:
        info = torch.cuda.memory_stats(self.device)
        max_active = info["active_bytes.all.peak"]
        max_reserved = info["reserved_bytes.all.peak"]
        num_retries = info["num_alloc_retries"]
        num_ooms = info["num_ooms"]

        if num_retries > 0:
            logger.warning(f"{num_retries} CUDA memory allocation retries.")
        if num_ooms > 0:
            logger.warning(f"{num_ooms} CUDA OOM errors thrown.")

        return GPUMemStats(
            max_active_gib=max_active / (1024**3),
            max_active_pct=100 * max_active / self.device_capacity,
            max_reserved_gib=max_reserved / (1024**3),
            max_reserved_pct=100 * max_reserved / self.device_capacity,
            num_alloc_retries=num_retries,
            num_ooms=num_ooms,
        )

    def reset_peak_stats(self) -> None:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

    def __str__(self) -> str:
        stats = self.get_peak_stats()
        return (
            f"{self.device_name} ({self.device_index}): "
            f"{self.device_capacity_gib:.2f}GiB capacity, "
            f"{stats.max_reserved_gib:.2f}GiB peak, "
            f"{stats.max_reserved_pct:.0f}% peak"
        )
