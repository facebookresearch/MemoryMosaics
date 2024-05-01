"""
Full definition of a Memory Mosaic Language Model, all of it in this single file.
This is intentionally kept as close as possible to the original gpt_model.py.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class LeakyAvg(nn.Module):
    def __init__(self, block_size, n_head):
        super().__init__()
        coef = torch.zeros(block_size, block_size)
        for i in range(block_size):
            coef = torch.diagonal_scatter(coef, -torch.ones(block_size-i)*i, -i)
        self.register_buffer('coef', coef)
        self.exp_scaling = 10
        self.leaky_key_beta = nn.Parameter(torch.linspace(0.5, 5, n_head).view(1, n_head, 1, 1)/self.exp_scaling)

    def forward(self, k):
        B, nh, T, hs = k.size()
        leaky_key_beta = self.leaky_key_beta.abs() * self.exp_scaling
        coef = self.coef[:T,:T].view(1,1,T,T)
        coef = torch.exp(coef * leaky_key_beta)
        return coef.tril() @ k

class KeyFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.leaky_cuda = config.leaky_cuda
        self.W_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.leaky_avg = LeakyAvg(config.block_size, config.n_head)
        self.exp_scaling = 10
        self.key_scale = nn.Parameter(torch.ones(1, config.n_head, 1, 1) / self.exp_scaling)
        self.key_scale_max = math.log(2**16-1) # fits in fp16.

    def forward(self, x, scale_pow=1):
        B,T,C = x.size()
        hs = C // self.n_head
        k = self.W_k(x).transpose(1,2).view(B, self.n_head, hs, T).transpose(2,3)
        k = self.leaky_avg(k)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-10)
        k = k * (scale_pow * self.exp_scaling * self.key_scale).clamp(max=self.key_scale_max).exp()
        return k

class ValFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        v_shift = 1 # access to x_T+1
        self.shift_fn = lambda x: F.pad(x, (-v_shift, v_shift))
        self.W_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.coef = nn.Parameter(torch.rand(1, config.n_head, 1, 1))
        self.exp_scaling = 10
        val_scale_init = -.5
        self.val_scale  = nn.Parameter(torch.ones(1, config.n_head, 1, 1) * val_scale_init / self.exp_scaling)

    def forward(self, x):
        B,T,C = x.size()
        hs = C // self.n_head
        v = self.W_v(x).transpose(1,2).view(B, self.n_head, hs, T)
        v = (1-self.coef) * self.shift_fn(v) + self.coef * v
        v = v.transpose(2,3)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-10)
        v = v * (self.exp_scaling * self.val_scale).exp()
        return v

class ContextMem(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, value projections for all heads, but in a batch
        self.k_featurizer = KeyFeatureExtractor(config)
        self.v_featurizer = ValFeatureExtractor(config)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # causal mask to ensure that attention is only applied to the left in the input sequence
        if not self.flash:
            self.attn_dropout = nn.Dropout(config.dropout)
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size), diagonal=-1)
                                    .view(1, 1, config.block_size, config.block_size).bool())

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate key, values for all heads in batch and move head forward to be the batch dim
        k = self.k_featurizer(x) # B, nh, T, hs
        v = self.v_featurizer(x) # B, nh, T, hs

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.zeros_like(v)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y[:, :, 1:] = torch.nn.functional.scaled_dot_product_attention(
                k[:, :, 1:],
                k,
                v,
                attn_mask=self.bias[:, :, 1:T, :T],
                dropout_p=self.dropout if self.training else 0,
                scale=1
            )
        else:
            # manual implementation of attention
            att = (k[:,:,1:] @ k.transpose(-2, -1))
            att = att.masked_fill(~self.bias[:,:,1:T,:T], float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y[:, :, 1:] = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class PersistentMem(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, value projections for all heads, but in a batch
        self.k_featurizer = KeyFeatureExtractor(config)
        pmem_dim = config.n_embd // config.n_head
        self.P_k = nn.Parameter(torch.zeros(config.pmem_count, 1, config.n_head, config.pmem_size, pmem_dim))
        self.P_v = nn.Parameter(torch.zeros(config.pmem_count, 1, config.n_head, config.pmem_size, pmem_dim))
        self.exp_scaling = 10
        out_scale_init = -.5
        self.out_scale  = nn.Parameter(torch.ones(1, config.n_head, 1, 1) * out_scale_init / self.exp_scaling)
        torch.nn.init.normal_(self.P_k, mean=0.0, std=1 / math.sqrt(pmem_dim))
        torch.nn.init.normal_(self.P_v, mean=0.0, std=1 / math.sqrt(pmem_dim))

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.pmem_count = config.pmem_count
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # causal mask to ensure that attention is only applied to the left in the input sequence
        if not self.flash:
            self.attn_dropout = nn.Dropout(config.ic_dropout)
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate key, values for all heads in batch and move head forward to be the batch dim
        k = self.k_featurizer(x, scale_pow=2) # 2 because P_k does not have scale

        if self.flash:
            y = 0
            for i in range(self.pmem_count):
                y = y + F.scaled_dot_product_attention(
                    k,
                    self.P_k[i],
                    self.P_v[i],
                    scale=1,
                    dropout_p=self.dropout if self.training else 0,
                )
        else:
            # manual implementation of attention
            for i in range(self.pmem_count):
                att = k @ (self.P_k[i].transpose(-2, -1))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y += att @ self.P_v[i] # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y / self.pmem_count
        y = y * (self.exp_scaling * self.out_scale).exp()
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = ContextMem(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = PersistentMem(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class MemoryMosaicConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    pmem_size: int = 2688
    pmem_count: int = 1
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    leaky_cuda: bool = False # True: use LeakyAverageCuda, False: use LeakyAvg

class MemoryMosaic(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.mosaic = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.mosaic.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.mosaic.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.mosaic.drop(tok_emb)
        for block in self.mosaic.h:
            x = block(x)
        x = self.mosaic.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.mosaic.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        leave_out = lambda x : x.dim() < 2 or x.shape[-2:] == (1, 1)
        decay_params = [p for n, p in param_dict.items() if not leave_out(p)]
        nodecay_params = [p for n, p in param_dict.items() if leave_out(p)]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
