# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""The compiled flex_attention every layer uses (uncompiled it materializes the full
score matrix)."""

import torch
from torch.nn.attention.flex_attention import flex_attention as _flex_attention

flex_attention = torch.compile(_flex_attention, dynamic=None)
# Evaluation and generation see many sequence lengths; let each compiled block keep a
# graph per shape instead of falling back to eager after the default 8.
torch._dynamo.config.cache_size_limit = 1024
