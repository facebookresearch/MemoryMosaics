# %%
import numpy as np
import mlx.core as mx
import mlx.nn as nn
#import mlx.optimizers as optim


# normalize
def normalize(x):
    """L2-normalize x along its last dim"""
    return x / (mx.linalg.norm(x, axis=-1, keepdims=True) + 1e-3)

# complexifu
def complexify(weight):
    ## pytorch possible
    #oops=FloatMatrix([[1,0,0,1],[0,-1,1,0],[0,1,-1,0],[1,0,0,1]]) * 0.5
    #wx=w.view(R/2,2,C/2,2).transpose(0,2,1,3).contiguous().reshape(R/2,C/2,4)
    #return (wx @ oops).reshape(R/2,C/2,2,2).transpose(0,2,1,3).contiguous().reshape(R,C)
    weight = weight * 0.5
    weight[0::2,0::2] += weight[1::2,1::2]
    weight[0::2,1::2] -= weight[1::2,0::2]
    weight[1::2,0::2] = -weight[0::2,1::2]
    weight[1::2,1::2] = weight[0::2,0::2]
    return weight

class ComplexLinear(nn.Linear):
    def __init__(self, idim, odim, bias=True):
        super().__init__(idim, odim, bias=bias)
        self.weight = complexify(self.weight)
    def __call__(self, x):
        weight = complexify(self.weight)
        if "bias" in self:
            x = mx.addmm(self.bias, x, weight.T)
        else:
            x = x @ weight.T
        return x

class ContextMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.h_dim = config.h_dim
        self.input_dim = config.input_dim
        self.v_shift = config.v_shift
        self.v_ban = config.v_ban
        self.norm_keys = config.norm_keys
        self.norm_vals = config.norm_vals
        self.k_scale = mx.array(1.0) if config.scale_keys else None
        self.attn_scale = config.attn_scale or 1
        edim = self.h_dim * self.n_head
        linear = ComplexLinear if config.complex else nn.Linear
        # initial embedding
        self.emb = linear(config.input_dim, edim, bias=False) if config.emb else None
        # additive mask
        t = config.block_size
        self.mask = mx.triu(mx.full((t,t), vals=-1e10), k=1-self.v_shift-self.v_ban)
        self.freeze(keys=['mask'],recurse=False,strict=True)
        # key extraction (linear)
        self.k_lin = linear(edim if config.emb else config.input_dim, edim, bias=False)
        self.kfun = lambda x: self.k_lin(x)
        # value extraction (linear)
        self.v_lin = linear(edim if config.emb else config.input_dim, edim, bias=False)
        self.vfun = lambda x: mx.pad(self.v_lin(x[:,self.v_shift:,:]),((0,0),(0,self.v_shift),(0,0)))
        # output projection
        self.out_proj = linear(edim, config.input_dim, bias=False) if config.out_proj else None
        self.out_sum = config.out_sum

    def __call__(self, x, cache=None):
        nh = self.n_head
        hd = self.h_dim
        B, T, C = x.shape
        assert C == self.input_dim
        if self.emb:
            x = self.emb(x)
        # extract key/values
        k = self.kfun(x).reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        if self.norm_keys:
            k = normalize(k)
        if not self.k_scale is None:
            k = k * mx.exp(self.k_scale)
        v = self.vfun(x).reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        if self.norm_vals:
            v = normalize(v)
        # memory attention
        att = k @ k.transpose(0,1,3,2) + self.mask[None,None,:T,:T]
        if cache:
            kc = mx.concatenate([cache[0], k], axis=-2)
            vc = mx.concatenate([cache[1], v], axis=-2)
            att = mx.concatenate([k @ cache[0].transpose(0,1,3,2), att], axis=-1)
            att = mx.softmax(att * self.attn_scale, axis=-1)
        else:
            kc = k
            vc = v
            sh = min(T, self.v_shift + self.v_ban)
            att = mx.softmax(att * self.attn_scale, axis=-1)
            att[:,:,:sh,:] = 0
        mx.eval(att,kc,vc)
        y = att @ vc # (B, nh, T, T') x (B, nh, T', hd) -> (B, nh, T, hd)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
        # final projection
        if self.out_proj:
            y = self.out_proj(y)
        elif self.out_sum:
            y = mx.sum(y.reshape(B, T, nh, hd), axis=-2) 
        return y, (kc, vc), att


    def generate(self, x, how_many):
        outs=[]
        out,cache,_ = self(x)
        mx.eval(out,cache)
        for _ in range(how_many):
            x = out[:,-1,:][:, None, :]
            outs.append(out)
            out,cache,_ = self(x, cache=cache)
            mx.eval(out,cache)
        outs.append(out)
        return mx.concatenate(outs, axis=1)

    def generate_nocache(self, x, how_many):
        for _ in range(how_many+1):
            out,_,_ = self(x)
            y = out[:,-1,:][:,None,:]
            x = mx.concatenate([x, y], axis=1)
        return out
