import math, os
import numpy as np
#import logging
import torch
import torch.nn as nn
#import inspect
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import subprocess
import time 
import warnings

FLOAT_MODE = 'fp16'
wkv_cuda = None 
T_MAX = 1024

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        #assert B * C % min(C, 1024) == 0
        global FLOAT_MODE 


        if '32' in FLOAT_MODE:
            w = w.contiguous()
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        else:
            w = w.float().contiguous()
            u = u.float().contiguous()
            k = k.float().contiguous()
            v = v.float().contiguous()
  

        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device="cuda", memory_format=torch.contiguous_format)
        #print(w.dtype, k.dtype, v.dtype,y.dtype)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        

        if "32" in FLOAT_MODE:
            return y
        elif FLOAT_MODE == "fp16":
            return y.half()
        elif FLOAT_MODE == "bf16":
            return y.bfloat16()

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        #assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors

        gw = torch.zeros((B, C), device="cuda").contiguous()
        gu = torch.zeros((B, C), device="cuda").contiguous()
        gk = torch.zeros((B, T, C), device="cuda").contiguous()
        gv = torch.zeros((B, T, C), device="cuda").contiguous()

        if "32" in FLOAT_MODE:
            wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        else:
            wkv_cuda.backward(
                B, T, C, w, u, k, v, gy.float().contiguous(), gw, gu, gk, gv
            )
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        #print(gw,gu,gk,gv)
        if "32" in FLOAT_MODE:
            return (None, None, None, gw, gu, gk, gv)
        elif FLOAT_MODE == "fp16":
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif FLOAT_MODE == "bf16":
            return (
                None,
                None,
                None,
                gw.bfloat16(),
                gu.bfloat16(),
                gk.bfloat16(),
                gv.bfloat16(),
            )


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


class LeakyAverageCuda(nn.Module):
    def __init__(self, n_embd, n_head, max_seq_length, leaky_per_head=True, sep_w_on_t = True, linear=False, bias=False):
        super().__init__()
        
        assert n_embd % n_head == 0
        self.leaky_per_head = leaky_per_head
        self.sep_w_on_t = sep_w_on_t

        self.n_embd = n_embd
        self.n_head = n_head
        self.leaky_per_head = leaky_per_head
        #self.max_seq_length = max_seq_length
        leaky_dim = self.n_head if self.leaky_per_head else self.n_embd

        self.leaky_beta_w_under  = nn.Parameter(torch.linspace(0.5,5,leaky_dim)/10)        
        self.leaky_beta_u_under  = nn.Parameter(torch.linspace(0.5,5,leaky_dim)/10) if self.sep_w_on_t else self.leaky_beta_w_under 
        

        # if self.leaky_per_head:
        #     self.leaky_beta_w = self.leaky_beta_w_under.view(self.n_head,1).repeat(1, self.n_embd // self.n_head ).flatten()
        #     self.leaky_beta_u = self.leaky_beta_u_under.view(self.n_head,1).repeat(1, self.n_embd // self.n_head ).flatten()
        # else:
        #     self.leaky_beta_w = self.leaky_beta_w_under
        #     self.leaky_beta_u = self.leaky_beta_u_under

        self.linear = nn.Linear(n_embd, n_embd, bias = bias) if linear else None  

        
        global T_MAX 
        T_MAX = max_seq_length
        rootdir = os.path.dirname(__file__)

        # cuda 11.4 has a bug, such that one need to use gcc version 10 instead of 11
        # https://forums.developer.nvidia.com/t/cuda-11-5-samples-throw-multiple-error-attribute-malloc-does-not-take-arguments/192750
        
        gccversion = subprocess.run(['gcc', '--version'], stdout=subprocess.PIPE)
        gccversion = gccversion.stdout.decode("utf-8").split('\n')[0].split(' ')[-1]
        
        cudaversion = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE)
        cudaversion = cudaversion.stdout.decode('utf-8').split('\n')[-3].split(' ')[-1]

        if '11.4' in gccversion and '11.4' in cudaversion:
            warnings.warn(f'\n gccversion is {gccversion}, cudaversion is {cudaversion}. They are not compatiable. \n Please try to use gcc-10 by the following command: \n export CC=/usr/bin/gcc-10 \n')
            


        global wkv_cuda
        if wkv_cuda is None:

            wkv_cuda = load(
                name="wkv",
                sources=[os.path.join(rootdir, "cuda/wkv_op.cpp"), os.path.join(rootdir, "cuda/wkv_cuda.cu")],
                verbose=True,
                extra_cuda_cflags=[
                ""
                    "-res-usage",
                    "--maxrregcount 60",
                    "--use_fast_math",
                    "-O3",
                    "-Xptxas -O3",
                    f"-DTmax={T_MAX}",
                ],
            )

    def forward(self, x, gate=None):
        B, T, C = x.size() 
        
        if self.linear is not None:
            x = self.linear(x)

        global FLOAT_MODE
        FLOAT_MODE = {torch.float32:'fp32', torch.bfloat16:"bf16",torch.float16:"fp16"}[x.dtype]
        #print(x.dtype, self.leaky_beta_w_under.device)
        #print(FLOAT_MODE)
        #print(self.linear.weight.dtype)
        
        gate = torch.zeros((B, T, C), device="cuda") if gate is None else gate # set k to zero to omit k
        #print(self.leaky_beta_w.device, self.leaky_beta_u.device, gate.device, x.device)
        if self.leaky_per_head:
            leaky_beta_w = self.leaky_beta_w_under.view(self.n_head,1).repeat(1, self.n_embd // self.n_head ).flatten()
            leaky_beta_u = self.leaky_beta_u_under.view(self.n_head,1).repeat(1, self.n_embd // self.n_head ).flatten()
        else:
            leaky_beta_w = self.leaky_beta_w_under
            leaky_beta_u = self.leaky_beta_u_under

        leaky_beta_w = -leaky_beta_w.abs()
        leaky_beta_u = leaky_beta_u.abs()

        output = RUN_CUDA(B, T, C, leaky_beta_w*10, leaky_beta_u*10, gate, x)
        
        #print(output.dtype)
        return output

    # def to(self, device, **kwargs): #todo currently, to() function does not work
    #     super().to(device, **kwargs)
    #     self.leaky_beta_w = self.leaky_beta_w.to(device,**kwargs)
    #     self.leaky_beta_u = self.leaky_beta_u.to(device,**kwargs)
    #     return self
