# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

from memory_mosaics.models import StackAssoMem as StackAssoMemBase
import math
from collections import namedtuple

class StackAssoMem(StackAssoMemBase):
    def __init__(self, vocab_size, n_head=12, n_embd=768, n_layer=12, v_shift=1, block_size=512, \
                k_fe_type='linearconv', v_fe_type='lowrlinearconv', \
                k_kernel_size=1, v_kernel_size=2, \
                ic_dropout=0.05, hd_dropout=0.05, 
                bias=False, 
                pmem_count=1, pmem_size=2688, pre_ln=True, weight_tying=True, skip_tokens=0, attn_only=False, \
                config={}):
        super().__init__(vocab_size=vocab_size, n_head=n_head, n_embd=n_embd, n_layer=n_layer, v_shift=v_shift, block_size=block_size, \
                k_fe_type=k_fe_type, v_fe_type=v_fe_type, \
                k_kernel_size=k_kernel_size, v_kernel_size=v_kernel_size, \
                ic_dropout=ic_dropout, hd_dropout=hd_dropout, 
                bias=bias, 
                pmem_count=pmem_count, pmem_size=pmem_size, pre_ln=pre_ln, weight_tying=weight_tying, skip_tokens=skip_tokens, attn_only=attn_only, \
                config=config)

        #to gpu 
        self = self.to('cuda')

    def forward(self, idx, state=None,return_hidden_outputs=None):#, futures=None, storyid=None, skip_next=False, skip_future=False):  # fugures (b,n,t)
       

        device = idx.device
        b, t = idx.size()
        
        x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)

        logits, loss =  None, 0
        logits_scale = 1.0 / math.sqrt(self.n_embd)

        logits = logits_scale * self.head(x[:,self.skip_tokens:])
            
        CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
        return CausalLMOutput(logits=logits), [], None

