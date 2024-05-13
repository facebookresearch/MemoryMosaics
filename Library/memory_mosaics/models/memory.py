# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

import torch 
import torch.nn  as nn
from torch.nn import functional as F
import numpy as np 

class Pmem(nn.Module):
    def __init__(self, pmem_count, pmem_size, pmem_dim, pmem_nhead, dropout ):
        super().__init__()
        self.pmem_size = pmem_size 
        self.pmem_dim = pmem_dim
        self.pmem_nhead = pmem_nhead
        self.pmem_count = pmem_count
        self.dropout = dropout
        self.M_k, self.M_v = [], []

        for i in range(self.pmem_count):
            M_k =  nn.Parameter(torch.zeros(1, self.pmem_nhead, self.pmem_size , self.pmem_dim)) 
            M_v = nn.Parameter(torch.zeros(1, self.pmem_nhead, self.pmem_size , self.pmem_dim)) 
            torch.nn.init.normal_(M_k, mean=0.0, std=1 / np.sqrt(self.pmem_dim))
            torch.nn.init.normal_(M_v, mean=0.0, std=1 / np.sqrt(self.pmem_dim))

            setattr(self, f'M_k{i}', M_k)
            setattr(self, f'M_v{i}', M_v)

            

    def forward(self, key): #key #(B, nh, T, hs)
        B, _, _, _ = key.size()
        y = 0
        for i in range(self.pmem_count): # TODO without expand, somethings there is a cuda memory error ill-memory
           y = y + F.scaled_dot_product_attention(key, (getattr(self, f'M_k{i}')).expand(B,self.pmem_nhead, self.pmem_size , self.pmem_dim ), \
                   (getattr(self, f'M_v{i}')).expand(B,self.pmem_nhead, self.pmem_size , self.pmem_dim ) ,is_causal=False, scale=1, dropout_p = self.dropout if self.training else 0)           
        y = y /  self.pmem_count

        return y 

