# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class StoriesDataset(Dataset):
    # randomly samply chunks as example
    def __init__(
        self, data_dir, block_size=512, split="train"):#, future_tokens=1, return_storyid=False):  # split \in [train, val]
        self.data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
        

        self.block_size = block_size
        #self.future_tokens = future_tokens

    def __len__(self):
        return len(self.data) - self.block_size - 2 # - self.future_tokens

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx : idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + 1 + self.block_size].astype(np.int64))

        return x, y 
        

