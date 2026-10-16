# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Configuration of the RNN family (`model_type: rnn`): DeltaNet."""

from dataclasses import dataclass

from model.common.config import BaseConfig


@dataclass
class RNNConfig(BaseConfig):
    n_heads: int | None = None            # None = dim // head_dim

    @property
    def heads(self) -> int:
        return self.n_heads if self.n_heads is not None else self.dim // self.head_dim
