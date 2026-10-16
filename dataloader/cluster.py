# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Cluster data loader.

Each step draws `clusters_per_step` slots (<= batch_size x world size), each a set
of clusters sampled in proportion to cluster size. Sequence b of data-parallel
rank r (global index g = r * batch_size + b) packs documents of slot
g % clusters_per_step, so each slot feeds batch_size x world size / clusters_per_step
sequences. The mix is controlled by `cluster_frac`: a dict mapping
{n_clusters: n_slots}. For example, {1: 28, 2: 3, 3: 1} means 28 slots hold 1
cluster, 3 slots 2 clusters, and 1 slot 3 clusters. The values must sum to
clusters_per_step. None defaults to {1: clusters_per_step}.

For multi-cluster slots, documents are pulled round-robin from each cluster's
pipeline (equal document count), shuffled at the document level with a seeded
RNG, then packed into one sequence.

Per-cluster pipelines are stateful (_ClusterDocPipeline), supporting exact
stop-and-resume via saved file byte offsets and shuffle-buffer RNG state. Each rank
reads its own byte-range share of every cluster file (about 1/world size of it), so
ranks fed by the same cluster never see the same document.
Batches are prefetched in a background process to hide I/O latency.
"""

import json
import logging
import multiprocessing as mp
import os
import pickle
import queue as _queue_mod
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from torch.distributed.checkpoint.stateful import Stateful

from tokenizer.tokenizer import Tokenizer
from dataloader.streaming import JsonlSource

logger = logging.getLogger()


class TrainBatch:
    __slots__ = ("x", "y", "mask", "src_names")

    def __init__(self, x, y, mask=None, src_names=None):
        self.x = x
        self.y = y
        self.mask = mask
        self.src_names = src_names


# ---------------------------------------------------------------------------
# Stateful per-cluster document pipeline
# ---------------------------------------------------------------------------

class _ClusterDocPipeline:
    """One cluster's endless stream of tokenized documents for this rank: its byte-range
    share of the cluster's JSONL files (dataloader.streaming.JsonlSource), shuffled
    through a buffer of raw texts, then tokenized. Tokens a sequence did not consume are
    pushed back (push_leftover) and come first next time. state_dict / load_state_dict
    resume it exactly; the state holds texts, which are compact (a trainer keeps the
    state of every cluster it has read)."""

    def __init__(
        self,
        cluster_dir: str,
        tokenizer,
        dp_rank: int,
        dp_world_size: int,
        seed: int,
        shuffle_buffer_size: int,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self._cluster_dir = cluster_dir
        self._tokenizer = tokenizer
        self._add_bos, self._add_eos = add_bos, add_eos
        self._source = JsonlSource(cluster_dir, dp_rank, dp_world_size)
        self._docs = (self._source.map(lambda raw: raw.get("text", raw.get("content")) or None)
                      .shuffle(shuffle_buffer_size, seed)
                      .map(self._tokenize))
        self._leftover_tokens: list[int] = []

    def _tokenize(self, text: str) -> dict:
        tokens = self._tokenizer.encode(text)
        return {"tokens": [self._tokenizer.bos_id] * self._add_bos + tokens
                + [self._tokenizer.eos_id] * self._add_eos}

    def next_doc(self) -> dict:
        """The next document (never ends: the cluster's files are read over and over)."""
        if self._leftover_tokens:
            tokens, self._leftover_tokens = self._leftover_tokens, []
            return {"tokens": tokens}
        try:
            return next(self._docs)
        except StopIteration:
            raise RuntimeError(f"No documents in {self._cluster_dir}") from None

    def push_leftover(self, tokens: list[int]) -> None:
        """Tokens a sequence did not consume; next_doc returns them first."""
        self._leftover_tokens = tokens

    def state_dict(self) -> dict:
        return {"docs": self._docs.state_dict(), "leftover_tokens": list(self._leftover_tokens)}

    def load_state_dict(self, state: dict) -> None:
        self._docs.load_state_dict(state["docs"])
        self._leftover_tokens = list(state["leftover_tokens"])

    def close(self) -> None:
        self._source.close()


# ---------------------------------------------------------------------------
# Packing helpers
# ---------------------------------------------------------------------------

def _pack_sequence(
    pipelines: list[_ClusterDocPipeline],
    seq_len: int,
    rng: random.Random,
) -> list[int]:
    """Pull docs round-robin from pipelines until seq_len+1 tokens, shuffle, pack.

    Tokens not consumed in this sequence (tail of a long doc, or entire docs
    shuffled past the cutoff) are pushed back to the originating pipeline so
    no tokens are wasted across sequence boundaries.

    Returns tokens: list[int] of length seq_len+1.
    """
    items: list[tuple[dict, int]] = []  # (doc, pipeline_idx)
    total_tokens = 0
    i = 0
    while total_tokens < seq_len + 1:
        pipe_idx = i % len(pipelines)
        doc = pipelines[pipe_idx].next_doc()
        items.append((doc, pipe_idx))
        total_tokens += len(doc["tokens"])
        i += 1

    if len(pipelines) > 1:
        rng.shuffle(items)

    tokens: list[int] = []
    # Accumulate per-pipeline leftovers; multiple items from the same pipeline
    # (possible with many-pipeline round-robin + shuffle) are concatenated in
    # encountered order so the pipeline returns them in sequence.
    pending: dict[int, list[int]] = {}

    for doc, pipe_idx in items:
        if len(tokens) >= seq_len + 1:
            # Entire doc not consumed — push back so it leads next time.
            pending[pipe_idx] = doc["tokens"] + pending.get(pipe_idx, [])
            continue
        prev_len = len(tokens)
        tokens.extend(doc["tokens"])
        if len(tokens) > seq_len + 1:
            # Tail of this doc overflows — push back the unused suffix.
            consumed = seq_len + 1 - prev_len
            tail = doc["tokens"][consumed:]
            if tail:
                pending[pipe_idx] = tail + pending.get(pipe_idx, [])

    for pipe_idx, leftover in pending.items():
        pipelines[pipe_idx].push_leftover(leftover)

    return tokens[:seq_len + 1]


# ---------------------------------------------------------------------------
# Background producer process
# ---------------------------------------------------------------------------

def _producer_process(
    queue: mp.Queue,
    stop_event: mp.Event,
    cluster_data_dir: str,
    schedule: list,          # list[list[list[int]]] — steps × batch × cluster_ids
    tokenizer_path: str,
    seq_len: int,
    dp_rank: int,
    dp_world_size: int,
    seed: int,
    max_cached_clusters: int,
    shuffle_buffer_size: int,
    add_bos: bool = True,
    add_eos: bool = True,
    start_step: int = 0,
    pipeline_states: dict | None = None,  # {cluster_id: state_dict} for resume
):
    # Any unhandled exception here kills the process silently (daemon=True).
    # Log it explicitly so it shows up in the training logs.
    try:
        _producer_process_inner(
            queue=queue, stop_event=stop_event,
            cluster_data_dir=cluster_data_dir, schedule=schedule,
            tokenizer_path=tokenizer_path,
            seq_len=seq_len, dp_rank=dp_rank, dp_world_size=dp_world_size,
            seed=seed, max_cached_clusters=max_cached_clusters,
            shuffle_buffer_size=shuffle_buffer_size,
            add_bos=add_bos, add_eos=add_eos,
            start_step=start_step, pipeline_states=pipeline_states,
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "Producer process crashed (dp_rank=%d, start_step=%d)", dp_rank, start_step
        )
        raise


def _producer_process_inner(
    queue: mp.Queue,
    stop_event: mp.Event,
    cluster_data_dir: str,
    schedule: list,
    tokenizer_path: str,
    seq_len: int,
    dp_rank: int,
    dp_world_size: int,
    seed: int,
    max_cached_clusters: int,
    shuffle_buffer_size: int,
    add_bos: bool = True,
    add_eos: bool = True,
    start_step: int = 0,
    pipeline_states: dict | None = None,
):
    tokenizer = Tokenizer(tokenizer_path)
    cache: OrderedDict[int, _ClusterDocPipeline] = OrderedDict()
    # The state of every pipeline not in `cache`: an evicted pipeline resumes where it
    # stopped, so the data stream does not depend on cache size or on resuming.
    parked: dict[int, dict] = dict(pipeline_states or {})

    def get_pipeline(cluster_id: int) -> _ClusterDocPipeline:
        if cluster_id in cache:
            cache.move_to_end(cluster_id)
            return cache[cluster_id]
        pipe = _ClusterDocPipeline(
            cluster_dir=os.path.join(cluster_data_dir, f"cluster_{cluster_id:04d}"),
            tokenizer=tokenizer,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            seed=seed + cluster_id * 1000,
            shuffle_buffer_size=shuffle_buffer_size,
            add_bos=add_bos,
            add_eos=add_eos,
        )
        if cluster_id in parked:
            pipe.load_state_dict(parked.pop(cluster_id))
        cache[cluster_id] = pipe
        while len(cache) > max_cached_clusters:
            evicted_id, evicted = cache.popitem(last=False)
            parked[evicted_id] = evicted.state_dict()
            evicted.close()
        return pipe

    doc_shuffle_rng = random.Random()

    for step in range(start_step, len(schedule)):
        if stop_event.is_set():
            break

        batch_cluster_lists = schedule[step]  # list of batch_size lists of cluster IDs
        x_list = []
        y_list = []
        src_names = []
        step_pipeline_states: dict[int, dict] = {}

        doc_shuffle_rng.seed(seed * 1_000_000_000 + step * 1000 + dp_rank)

        for cluster_ids in batch_cluster_lists:
            pipelines = [get_pipeline(int(cid)) for cid in cluster_ids]
            tokens = _pack_sequence(pipelines, seq_len, doc_shuffle_rng)

            x_list.append(np.array(tokens[:-1], dtype=np.int64))
            y_list.append(np.array(tokens[1:], dtype=np.int64))
            src_names.append("+".join(f"cluster_{cid}" for cid in cluster_ids))

            # After the sequence: a cluster used again later in the step is
            # overwritten, so the step ends with each cluster's final state.
            for pipe, cid in zip(pipelines, cluster_ids):
                step_pipeline_states[int(cid)] = pipe.state_dict()

        x = np.stack(x_list)
        y = np.stack(y_list)

        while not stop_event.is_set():
            try:
                queue.put(
                    (x, y, src_names, step_pipeline_states),
                    timeout=1.0,
                )
                break
            except Exception:
                continue


# ---------------------------------------------------------------------------
# ClusterDataIterator
# ---------------------------------------------------------------------------

class ClusterDataIterator(Stateful):
    def __init__(
        self,
        cluster_data_dir: str,
        *,
        tokenizer,
        seq_len: int,
        batch_size: int,
        total_steps: int,
        dp_rank: int,
        dp_world_size: int,
        seed: int,
        clusters_per_step: int = 0,
        cluster_frac: dict | None = None,
        prefetch_size: int = 16,
        max_cached_clusters: int = 256,
        shuffle_buffer_size: int = 64,
        cluster_metadata: str = "metadata.json",
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.cluster_data_dir = cluster_data_dir
        self.seq_len = seq_len
        self.batch_size = batch_size

        with open(os.path.join(cluster_data_dir, cluster_metadata)) as f:
            metadata = json.load(f)
        self.K = metadata["K"]
        self.cluster_sizes = metadata["cluster_sizes"]
        self.cluster_ids = metadata.get("cluster_ids", None)

        weights = np.array(self.cluster_sizes, dtype=np.float64)
        self.cluster_weights = weights / weights.sum()

        if max_cached_clusters <= 0:
            max_cached_clusters = self.K

        total_seqs = batch_size * dp_world_size
        if clusters_per_step <= 0:
            clusters_per_step = total_seqs
        clusters_per_step = min(clusters_per_step, total_seqs)

        # Resolve cluster_frac → {n_clusters: n_gpu_slots}
        if cluster_frac is None:
            cluster_frac = {1: clusters_per_step}
        assert sum(cluster_frac.values()) == clusters_per_step, (
            f"cluster_frac values must sum to clusters_per_step={clusters_per_step}, "
            f"got {sum(cluster_frac.values())}"
        )

        schedule = self._build_schedule(
            total_steps, batch_size, dp_rank, dp_world_size,
            clusters_per_step, cluster_frac, seed,
        )

        logger.info(
            f"ClusterDataIterator: {total_seqs} seqs/step, "
            f"clusters_per_step={clusters_per_step}, cluster_frac={cluster_frac}, "
            f"batch_size={batch_size}"
        )

        self._step = 0
        self._total_steps = total_steps
        self._schedule = schedule
        self._latest_pipeline_states: dict[int, dict] = {}
        self._dp_rank, self._dp_world_size = dp_rank, dp_world_size
        self._producer_pre_forked = False  # set by prepare_resume()

        self._prefetch_size = prefetch_size
        self._producer_kwargs = dict(
            cluster_data_dir=cluster_data_dir,
            schedule=schedule,
            tokenizer_path=tokenizer.tokenizer_path,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            seed=seed,
            max_cached_clusters=max_cached_clusters,
            shuffle_buffer_size=shuffle_buffer_size,
            add_bos=add_bos,
            add_eos=add_eos,
        )
        self._queue = mp.Queue(maxsize=prefetch_size)
        self._stop_event = mp.Event()
        self._producer = None
        self._start_producer(start_step=0, pipeline_states=None)

    def _build_schedule(
        self,
        total_steps: int,
        batch_size: int,
        dp_rank: int,
        dp_world_size: int,
        clusters_per_step: int,
        cluster_frac: dict,
        seed: int,
    ) -> list:
        """Build per-GPU schedule as list[list[list[int]]]: steps × batch × cluster_ids."""
        # Build the sizes list: one entry per GPU slot in [0, clusters_per_step)
        sizes: list[int] = []
        for n_clusters, n_slots in sorted(cluster_frac.items()):
            sizes.extend([n_clusters] * n_slots)
        # clusters_per_step == len(sizes) is guaranteed by __init__ assertion

        total_needed = sum(sizes)  # cluster IDs to sample per step globally

        rng = np.random.RandomState(seed)
        # Sample local indices in [0, K) — shape (total_steps, total_needed)
        global_ids = rng.choice(
            self.K, size=(total_steps, total_needed),
            replace=True, p=self.cluster_weights,
        )
        if self.cluster_ids is not None:
            id_map = np.array(self.cluster_ids, dtype=np.int64)
            global_ids = id_map[global_ids]

        # Global sequence g = dp_rank * batch_size + b takes slot g % clusters_per_step,
        # so every slot is used (clusters_per_step <= batch_size * dp_world_size) and
        # each slot feeds total_seqs / clusters_per_step sequences.
        starts = np.concatenate([[0], np.cumsum(sizes)])
        slots = [(dp_rank * batch_size + b) % clusters_per_step for b in range(batch_size)]
        per_seq = [global_ids[:, starts[s]:starts[s + 1]].tolist() for s in slots]
        return [[list(map(int, ids[step])) for ids in per_seq] for step in range(total_steps)]

    def __iter__(self) -> Iterator[TrainBatch]:
        while self._step < self._total_steps:
            # Use a timeout so a silently-dead producer raises a clear error
            # instead of hanging forever.
            while True:
                try:
                    payload = self._queue.get(timeout=60)
                    break
                except _queue_mod.Empty:
                    if self._producer is not None and not self._producer.is_alive():
                        raise RuntimeError(
                            f"Producer process died (exitcode={self._producer.exitcode}) "
                            f"at step {self._step}"
                        )
            x, y, src_names, pipeline_states = payload
            self._latest_pipeline_states.update(pipeline_states)
            batch = TrainBatch(x=x, y=y, mask=None, src_names=src_names)
            self._step += 1
            yield batch

    def _start_producer(self, start_step: int, pipeline_states: dict | None) -> None:
        self._stop_event = mp.Event()
        self._producer = mp.Process(
            target=_producer_process,
            kwargs=dict(
                queue=self._queue,
                stop_event=self._stop_event,
                start_step=start_step,
                pipeline_states=pipeline_states,
                **self._producer_kwargs,
            ),
            daemon=True,
        )
        self._producer.start()

    def state_dict(self) -> dict[str, Any]:
        # Only the step is stored via DCP (non-tensor values are only saved from
        # rank 0 in DCP, so per-rank pipeline states must be saved separately via
        # save_extra / load_extra which write per-rank pickle files).
        return {"step": self._step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._step = state_dict.get("step", 0)
        if self._producer_pre_forked:
            # prepare_resume() already killed the old producer and started the
            # right one before NCCL was initialised — nothing to do here.
            return
        # Normal path (no prepare_resume): kill old producer, fresh queue.
        # load_extra() will start the new producer.
        if self._producer is not None:
            self._producer.kill()
            self._producer = None
        self._queue = mp.Queue(maxsize=self._prefetch_size)
        self._latest_pipeline_states = {}

    def prepare_resume(self, ckpt_dir: str | Path) -> None:
        """Fork the producer from the checkpoint step BEFORE dcp.load (i.e. before
        NCCL is initialised).  Call this from train.py immediately before
        load_checkpoint().  Forking after NCCL init is unsafe because NCCL
        creates background threads; their mutexes are inherited locked by the
        child, causing silent deadlocks.

        The step is read from the checkpoint directory name (``step_N``).
        Pipeline states are loaded from the per-rank pickle written by
        save_extra().
        """
        ckpt_dir = Path(ckpt_dir)
        try:
            step = int(ckpt_dir.name.split("_")[1])
        except (IndexError, ValueError):
            logger.warning(f"prepare_resume: cannot parse step from {ckpt_dir.name}, skipping")
            return

        pipeline_states = self._read_states(ckpt_dir)

        # Kill the step-0 producer started in __init__ and start a new one.
        if self._producer is not None:
            self._producer.kill()
            self._producer = None
        self._queue = mp.Queue(maxsize=self._prefetch_size)
        self._step = step
        self._latest_pipeline_states = pipeline_states or {}
        self._start_producer(start_step=step, pipeline_states=pipeline_states)
        self._producer_pre_forked = True
        logger.info(f"prepare_resume: producer forked at step={step} (before NCCL init)")

    def _states_path(self, path: str | Path) -> Path:
        return Path(path) / f"pipeline_states_dp{self._dp_rank}.pkl"

    def _read_states(self, path: str | Path) -> dict | None:
        """This rank's saved pipeline states, {cluster_id: state}, or None (restart)."""
        pkl_path = self._states_path(path)
        if not pkl_path.exists():
            logger.warning(f"No pipeline states at {pkl_path}; pipelines restart from the beginning")
            return None
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)
        if not isinstance(saved, dict) or "dp_world_size" not in saved:
            logger.warning(f"{pkl_path} was written by an earlier version of the data loader; "
                           f"pipelines restart from the beginning")
            return None
        if saved["dp_world_size"] != self._dp_world_size:
            raise RuntimeError(
                f"{pkl_path} was saved by {saved['dp_world_size']} data-parallel ranks, this run has "
                f"{self._dp_world_size}: each rank's data position is inside its own share of the files, "
                f"so it cannot resume. Delete the checkpoint's pipeline_states_dp*.pkl to restart the "
                f"data from the beginning.")
        logger.info(f"Loaded pipeline states for {len(saved['clusters'])} clusters from {pkl_path}")
        return saved["clusters"]

    def save_extra(self, path: str | Path) -> None:
        """Save per-rank pipeline states to a pickle file alongside the DCP checkpoint."""
        pkl_path = self._states_path(path)
        with open(pkl_path, "wb") as f:
            pickle.dump({"dp_world_size": self._dp_world_size, "clusters": self._latest_pipeline_states}, f)
        logger.info(f"Saved pipeline states for {len(self._latest_pipeline_states)} clusters to {pkl_path}")

    def load_extra(self, path: str | Path) -> None:
        """Load per-rank pipeline states and (re)start the producer."""
        if self._producer_pre_forked:
            # prepare_resume() already handled everything before NCCL init.
            self._producer_pre_forked = False
            return
        pipeline_states = self._read_states(path)
        self._latest_pipeline_states = pipeline_states or {}
        self._start_producer(start_step=self._step, pipeline_states=pipeline_states)

    def close(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._producer is not None:
            self._producer.join(timeout=5)
            if self._producer.is_alive():
                self._producer.kill()
            self._producer = None
