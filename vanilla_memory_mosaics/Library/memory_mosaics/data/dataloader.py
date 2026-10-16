# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

import torch
from torch.utils.data import Dataset


# copied from https://github.com/facebookresearch/DomainBed/blob/main/domainbed/lib/fast_data_loader.py
class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


# modified from https://github.com/facebookresearch/DomainBed/blob/main/domainbed/lib/fast_data_loader.py
class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, num_workers=1):
        super().__init__()

        self.batch_size = batch_size

        sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=batch_size, drop_last=True
        )

        self._infinite_iterator = iter(
            torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=_InfiniteSampler(batch_sampler),
            )
        )

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError
