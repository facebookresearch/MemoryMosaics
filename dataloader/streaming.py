# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Composable, resumable streaming pipelines.

A pipeline is a chain of stages, each an iterator over the items of the one before:

    pipeline = (
        JsonlSource("/data/train", dp_rank=0, dp_world_size=8)   # endless: wraps around
        .map(tokenize)                     # None results are dropped
        .shuffle(buffer_size=64, seed=42)
        .pack(seq_len=4097, wrap=True)
        .batch(batch_size=4)
    )
    batch = next(pipeline)

Every stage has `state_dict()` / `load_state_dict()`, covering the stages before it: a
pipeline rebuilt with the same arguments and loaded with a saved state continues
exactly where the saved one stopped. Items are dicts, {"tokens": list[int],
"mask": list[bool]} once tokenized.

Sharding across data-parallel ranks is by byte range: rank r of W reads the lines that
START in bytes [r S / W, (r+1) S / W) of each file of size S. Every line goes to exactly
one rank, and each rank reads only its own share (about 1/W of the bytes).
"""

import json
import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger()


def jsonl_files(path: str) -> list[str]:
    """The .jsonl / .json files under `path`, recursively, in sorted order."""
    return sorted(
        str(Path(root) / f)
        for root, _, files in os.walk(path, followlinks=True)
        for f in files if f.endswith(".jsonl") or f.endswith(".json")
    )


class Stage:
    """An iterator with a resumable state; the fluent methods add stages after it."""

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError

    def map(self, fn: Callable) -> "Stage":
        return Map(self, fn)

    def filter(self, fn: Callable) -> "Stage":
        return Map(self, lambda item: item if fn(item) else None)

    def shuffle(self, buffer_size: int, seed: int = 42) -> "Stage":
        return Shuffle(self, buffer_size, seed)

    def pack(self, seq_len: int, *, wrap: bool = True, pad: int = -1) -> "Stage":
        return Pack(self, seq_len, wrap, pad)

    def batch(self, batch_size: int) -> "Stage":
        return Batch(self, batch_size)


class JsonlSource(Stage):
    """The JSON objects of the JSONL files under `path`, this rank's byte-range share of
    each file, file after file, starting over at the end (repeat=True) or stopping.
    Blank lines and lines that are not valid JSON are skipped.

    A rank whose share holds no line at all (a source with fewer lines than ranks)
    reads every line instead, after a warning: duplicated data rather than a hang."""

    def __init__(self, path: str, dp_rank: int = 0, dp_world_size: int = 1, repeat: bool = True) -> None:
        self.path, self.repeat = path, repeat
        self._rank, self._world = dp_rank, dp_world_size
        self._files = jsonl_files(path)
        if not self._files:
            logger.warning(f"No JSONL files found in {path}")
        self._file = None             # open handle of _files[_file_idx]
        self._end = 0                 # lines starting at or after this byte belong to the next rank
        self._file_idx = 0
        self._offset: int | None = None   # where the next line starts; None: at the share's start
        self._epoch = 0
        self._epoch_empty = True      # no line read yet in this epoch
        self._whole = False           # read every line (this rank's share was empty)

    def _share(self, size: int) -> tuple[int, int]:
        if self._whole:
            return 0, size
        return size * self._rank // self._world, size * (self._rank + 1) // self._world

    def _open(self) -> bool:
        """Open the current file at the next line to read; False if it is unreadable."""
        path = self._files[self._file_idx]
        try:
            self._file = open(path, "rb")
        except (FileNotFoundError, PermissionError) as e:
            logger.warning(f"Skipping {path}: {e}")
            return False
        start, self._end = self._share(os.fstat(self._file.fileno()).st_size)
        if self._offset is not None:
            self._file.seek(self._offset)
        elif start > 0:
            # The first line starting at or after `start`: skip the rest of the line
            # holding byte start - 1 (only its newline, if a line starts at `start`).
            self._file.seek(start - 1)
            self._file.readline()
        return True

    def _next_file(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file, self._offset = None, None
        self._file_idx += 1

    def __next__(self) -> dict:
        while True:
            if self._file_idx >= len(self._files):
                if not self.repeat or not self._files:
                    raise StopIteration
                if self._epoch_empty:
                    if self._whole or self._world == 1:
                        raise RuntimeError(f"No documents in {self.path}")
                    logger.warning(f"Rank {self._rank}/{self._world}: no line of {self.path} starts in "
                                   f"its share; it reads the whole source instead")
                    self._whole = True
                self._file_idx, self._epoch, self._epoch_empty = 0, self._epoch + 1, True
            if self._file is None and not self._open():
                self._next_file()
                continue
            if self._file.tell() >= self._end:
                self._next_file()
                continue
            line = self._file.readline()
            if not line:
                self._next_file()
                continue
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._epoch_empty = False
            return item

    def state_dict(self) -> dict:
        offset = self._file.tell() if self._file is not None else self._offset
        return {"file_idx": self._file_idx, "offset": offset, "epoch": self._epoch,
                "epoch_empty": self._epoch_empty, "whole": self._whole}

    def load_state_dict(self, state: dict) -> None:
        if self._file is not None:
            self._file.close()
        self._file, self._end = None, 0
        self._file_idx, self._offset = state["file_idx"], state["offset"]
        self._epoch, self._epoch_empty, self._whole = state["epoch"], state["epoch_empty"], state["whole"]

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class Map(Stage):
    """fn applied to every item; items it maps to None are dropped."""

    def __init__(self, parent: Stage, fn: Callable) -> None:
        self.parent, self.fn = parent, fn

    def __next__(self):
        while True:
            result = self.fn(next(self.parent))
            if result is not None:
                return result

    def state_dict(self) -> dict:
        return self.parent.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.parent.load_state_dict(state)


class Shuffle(Stage):
    """Buffer shuffle: keeps `buffer_size` items and returns a uniformly random one each
    time, refilling from the parent. Flushes the buffer in random order at the end."""

    def __init__(self, parent: Stage, buffer_size: int, seed: int = 42) -> None:
        self.parent, self.buffer_size = parent, buffer_size
        self._rng = np.random.RandomState(seed)
        self._buf: list = []
        self._flushing = False

    def __next__(self):
        if not self._flushing:
            try:
                while len(self._buf) < self.buffer_size:
                    self._buf.append(next(self.parent))
            except StopIteration:
                self._flushing = True
                self._rng.shuffle(self._buf)
        if self._flushing:
            if not self._buf:
                raise StopIteration
            return self._buf.pop(0)
        idx = self._rng.randint(len(self._buf))
        item = self._buf[idx]
        self._buf[idx] = self._buf[-1]
        self._buf.pop()
        return item

    def state_dict(self) -> dict:
        return {"rng": self._rng.get_state(), "buf": list(self._buf), "flushing": self._flushing,
                "parent": self.parent.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self._rng.set_state(state["rng"])
        self._buf, self._flushing = list(state["buf"]), state["flushing"]
        self.parent.load_state_dict(state["parent"])


class Pack(Stage):
    """Fixed-length sequences of `seq_len` tokens from tokenized items.

    wrap=True (pretraining): documents flow across sequence boundaries.
    wrap=False (SFT): an item never spans two sequences. One that does not fit in the
        space left ends the sequence, padded with `pad` (mask False), and starts the
        next; one longer than seq_len on its own is truncated.
    """

    def __init__(self, parent: Stage, seq_len: int, wrap: bool = True, pad: int = -1) -> None:
        self.parent, self.seq_len, self.wrap, self.pad = parent, seq_len, wrap, pad
        # The item being consumed, from `_offset` on. A sequence is built within one
        # __next__ call, so this is all the state between calls.
        self._item: dict | None = None
        self._offset = 0

    def __next__(self) -> dict:
        out_tokens: list[int] = []
        out_mask: list[bool] = []
        while True:
            if self._item is None:
                self._item, self._offset = next(self.parent), 0
            tokens, mask = self._item["tokens"], self._item["mask"]
            while self._offset < len(tokens):
                space = self.seq_len - len(out_tokens)
                remaining = len(tokens) - self._offset
                if not self.wrap and out_tokens and remaining > space:
                    return {"tokens": out_tokens + [self.pad] * space, "mask": out_mask + [False] * space}
                take = min(remaining, space)
                out_tokens += tokens[self._offset:self._offset + take]
                out_mask += mask[self._offset:self._offset + take]
                self._offset = len(tokens) if not self.wrap else self._offset + take
                if len(out_tokens) == self.seq_len:
                    return {"tokens": out_tokens, "mask": out_mask}
            self._item = None

    def state_dict(self) -> dict:
        return {"item": self._item, "offset": self._offset, "parent": self.parent.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self._item, self._offset = state["item"], state["offset"]
        self.parent.load_state_dict(state["parent"])


class Batch(Stage):
    """Lists of `batch_size` consecutive items."""

    def __init__(self, parent: Stage, batch_size: int) -> None:
        self.parent, self.batch_size = parent, batch_size

    def __next__(self) -> list:
        return [next(self.parent) for _ in range(self.batch_size)]

    def state_dict(self) -> dict:
        return self.parent.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.parent.load_state_dict(state)


class Sample(Stage):
    """Weighted interleaving: each item comes from source i with probability w_i / sum(w).
    The sources are endless (repeating) pipelines."""

    def __init__(self, sources: list[Stage], weights: list[float], seed: int = 42) -> None:
        self.sources = sources
        w = np.array(weights, dtype=np.float64)
        self._p = w / w.sum()
        self._rng = np.random.RandomState(seed)

    def __next__(self):
        return next(self.sources[self._rng.choice(len(self.sources), p=self._p)])

    def state_dict(self) -> dict:
        return {"rng": self._rng.get_state(), "sources": [s.state_dict() for s in self.sources]}

    def load_state_dict(self, state: dict) -> None:
        self._rng.set_state(state["rng"])
        for s, st in zip(self.sources, state["sources"], strict=True):
            s.load_state_dict(st)
