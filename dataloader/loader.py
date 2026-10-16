# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Data iterator dispatcher."""

import logging
import pickle
from pathlib import Path

import numpy as np
from torch.distributed.checkpoint.stateful import Stateful

from dataloader.cluster import ClusterDataIterator, TrainBatch
from dataloader.pretrain import build_pretrain_pipeline

logger = logging.getLogger()


class StreamIterator(Stateful):
    """TrainBatches from a resumable pipeline (pretrain, SFT).

    Its position differs on every rank, and DCP stores non-tensor state from rank 0 only,
    so the pipeline state is written beside the checkpoint, one file per rank
    (save_extra / load_extra); DCP keeps the batch count, which the file must match.
    """

    def __init__(self, pipeline, dp_rank: int, dp_world_size: int, with_mask: bool) -> None:
        self._pipeline = pipeline
        self._dp_rank, self._dp_world_size = dp_rank, dp_world_size
        self._with_mask = with_mask
        self._batches = 0

    def __iter__(self):
        return self

    def __next__(self) -> TrainBatch:
        items = next(self._pipeline)
        self._batches += 1
        tokens = np.array([item["tokens"] for item in items], dtype=np.int64)
        # mask[i + 1]: whether to compute the loss of predicting tokens[i + 1].
        mask = (np.array([item["mask"] for item in items], dtype=np.float32)[:, 1:]
                if self._with_mask else None)
        return TrainBatch(x=tokens[:, :-1], y=tokens[:, 1:], mask=mask)

    def state_dict(self) -> dict:
        return {"batches": self._batches}

    def load_state_dict(self, state_dict: dict) -> None:
        self._batches = state_dict.get("batches", 0)

    def _extra_path(self, path) -> Path:
        return Path(path) / f"data_state_dp{self._dp_rank}.pkl"

    def save_extra(self, path) -> None:
        with open(self._extra_path(path), "wb") as f:
            pickle.dump({"batches": self._batches, "dp_world_size": self._dp_world_size,
                         "pipeline": self._pipeline.state_dict()}, f)

    def load_extra(self, path) -> None:
        pkl = self._extra_path(path)
        if not pkl.exists():
            logger.warning(f"No data loader state at {pkl}; the data restarts from the beginning")
            return
        with open(pkl, "rb") as f:
            state = pickle.load(f)
        if not isinstance(state, dict) or "dp_world_size" not in state:
            logger.warning(f"{pkl} was written by an earlier version of the data loader; "
                           f"the data restarts from the beginning")
            return
        assert state["batches"] == self._batches, \
            f"{pkl} is at batch {state['batches']}, the checkpoint at batch {self._batches}"
        if state["dp_world_size"] != self._dp_world_size:
            raise RuntimeError(
                f"{pkl} was saved by {state['dp_world_size']} data-parallel ranks, this run has "
                f"{self._dp_world_size}: each rank's data position is inside its own share of the files, "
                f"so it cannot resume. Delete the checkpoint's data_state_dp*.pkl to restart the data "
                f"from the beginning.")
        self._pipeline.load_state_dict(state["pipeline"])
        logger.info(f"Data loader resumes after {self._batches} batches ({pkl})")

    def close(self) -> None:
        pass


def _parse_sources(root_dir: str, sources: dict[str, float]) -> list[tuple[str, float]]:
    root = Path(root_dir)
    return [(str(root / name), weight) for name, weight in sources.items()]


def build_data_iterator(args, tokenizer, dp_rank: int, dp_world_size: int):
    if args.data.loader == "pretrain":
        logger.info("Using pretrain streaming data loader")
        sources = _parse_sources(args.data.root_dir, args.data.sources)
        pipeline = build_pretrain_pipeline(
            sources=sources,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            seed=args.data.seed,
            shuffle_buffer_size=args.data.shuffle_buffer_size,
            add_bos=args.data.add_bos,
            add_eos=args.data.add_eos,
        )
        return StreamIterator(pipeline, dp_rank, dp_world_size, with_mask=False)

    elif args.data.loader == "cluster":
        logger.info("Using cluster data loader")
        assert args.data.cluster_data_dir is not None, \
            "cluster_data_dir must be set when loader='cluster'"
        return ClusterDataIterator(
            cluster_data_dir=args.data.cluster_data_dir,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            total_steps=args.steps,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            seed=args.data.seed,
            clusters_per_step=args.data.clusters_per_step,
            cluster_frac=args.data.cluster_frac,
            prefetch_size=args.data.prefetch_size,
            max_cached_clusters=args.data.max_cached_clusters,
            shuffle_buffer_size=args.data.shuffle_buffer_size,
            cluster_metadata=args.data.cluster_metadata,
            add_bos=args.data.add_bos,
            add_eos=args.data.add_eos,
        )

    elif args.data.loader == "sft":
        assert args.data.sft_data_path is not None, \
            "sft_data_path must be set when loader='sft'"
        from dataloader.sft import build_sft_pipeline
        pipeline = build_sft_pipeline(
            path=args.data.sft_data_path,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            seed=args.data.seed,
            shuffle_buffer_size=args.data.shuffle_buffer_size,
        )
        return StreamIterator(pipeline, dp_rank, dp_world_size, with_mask=True)

    else:
        raise ValueError(f"Unknown data loader: {args.data.loader}")
