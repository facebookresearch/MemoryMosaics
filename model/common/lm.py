# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""The language model around the blocks, shared by every family:
embedding -> n_layers blocks -> norm -> output projection (-> cross-entropy).

A family supplies the blocks and `build_masks`. Each step:

    mask = model.create_attention_mask(tokens, doc_lens)    # records the document layout
    loss = model(tokens, mask, targets=targets)             # or logits

Generation passes a cache (create_kv_cache), prefilling the prompt and then decoding one
token per step; a forward pass without a cache computes exactly as in training.
"""

import torch
import torch.nn.functional as F
from torch import nn

from model.common.config import ATTN_BIAS_TYPES, BaseConfig
from model.common.cross_entropy import fused_matmul_cross_entropy, weighted_mean_loss
from model.common.init import layer_init_factor
from model.common.mask import set_block_mask_compile
from model.common.norm import RMSNorm
from model.common.positions import DocumentLayout


def _cross_entropy(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.nll_loss(F.log_softmax(pred, -1, dtype=torch.float32), labels, reduction="none")


class LanguageModel(nn.Module):
    def __init__(self, args: BaseConfig, layers: list[nn.Module]) -> None:
        super().__init__()
        assert args.vocab_size > 0
        self.args = args
        self.attn_bias_type = args.attn_bias_type
        self.vocab_size = args.vocab_size
        self.max_seq_len = args.max_seq_len        # the training length
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        if args.weight_tying:
            self.output.weight = self.tok_embeddings.weight
        self.loss_fn = None if args.fused_cross_entropy else _cross_entropy
        self.layout = DocumentLayout()

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def blocks(self) -> list[nn.Module]:
        """The blocks, unwrapped from torch.compile."""
        return [getattr(layer, "_orig_mod", layer) for layer in self.layers]

    # ── masks ──

    def build_masks(self, tokens: torch.Tensor, q_lens: torch.Tensor, kv_lens, attn_bias_type: str,
                    step: int, seed: int) -> dict:
        """{mask name: FlexAttentionBias} for one full-sequence step; see the families."""
        raise NotImplementedError

    def create_attention_mask(
        self,
        tokens: torch.Tensor,
        seq_lens: torch.Tensor | list[int] | None = None,
        kv_seq_lens: torch.Tensor | list[int] | None = None,
        start_pos: int | None = None,
        attn_bias_type: str | None = None,
        step: int = 0,
        seed: int = 0,
        compile_mask: bool = True,
    ) -> dict | None:
        """Masks for `tokens` (bsz, seq_len) with document lengths `seq_lens` (row-major),
        and the document layout forward needs. A decoding step (seq_len 1, at position
        `start_pos`) has no masks and returns None. `attn_bias_type` overrides the
        model's (e.g. to evaluate a document-wise model on running text)."""
        attn_bias_type = attn_bias_type or self.attn_bias_type
        assert attn_bias_type in ATTN_BIAS_TYPES, f"attn_bias_type={attn_bias_type!r}"
        bsz, seq_len = tokens.shape
        # Compile block-mask creation for fixed-shape training only; the varying lengths
        # of generation and evaluation would recompile it every time.
        set_block_mask_compile(compile_mask)
        # Grow RoPE tables here, outside compiled code (see Attention.ensure_freqs_cis).
        for module in self.modules():
            if hasattr(module, "ensure_freqs_cis"):
                module.ensure_freqs_cis(max(seq_len, (start_pos or 0) + 1), tokens.device)
        if seq_len <= 1:
            self.layout.decoding(attn_bias_type, start_pos or 0, bsz)
            return None
        as_tensor = (lambda lens: torch.tensor(lens, device=tokens.device)
                     if isinstance(lens, list) else lens)
        q_lens = as_tensor(seq_lens)
        self.layout.full_sequence(attn_bias_type, q_lens, seq_len)
        return self.build_masks(tokens, q_lens, as_tensor(kv_seq_lens), attn_bias_type, step, seed)

    # ── generation ──

    def create_kv_cache(self, bsz: int, seq_length: int, dtype: torch.dtype,
                        device: str | torch.device = "cuda") -> list:
        """Empty per-layer generation caches for up to `seq_length` positions."""
        self.layout.reset()
        return [block.init_cache(bsz, seq_length, dtype, device) for block in self.blocks()]

    def end_documents(self, cache: list, rows: torch.Tensor) -> None:
        """Under "doc_causal", rows flagged in `rows` (bool [bsz]) just read an
        end-of-document token, so their next token starts a new document: positions
        restart, in-document memories stop reading the old document, cluster memories
        start reading it, and recurrent states reset. The generator calls this after
        every step with rows = (last input token == EOS)."""
        if not self.layout.documentwise:
            return
        self.layout.end_documents(rows)

        def reset(c):
            if hasattr(c, "reset_rows"):
                c.reset_rows(rows)
            elif hasattr(c, "__dataclass_fields__"):
                for value in vars(c).values():
                    reset(value)
        for layer_cache in cache:
            reset(layer_cache)

    # ── forward ──

    def forward(
        self,
        tokens: torch.Tensor,
        mask: dict | None,
        *,
        targets: torch.Tensor | None = None,
        targets_mask: torch.Tensor | None = None,
        cache: list | None = None,
        return_hidden: bool = False,
    ):
        """Mean next-token loss if `targets` is given (`targets_mask` restricts it, e.g.
        to answers in SFT), else logits -- and the updated cache when generating."""
        bsz, seq_len = tokens.shape
        assert cache is None or targets is None
        with torch.no_grad():
            pos = self.layout.positions(tokens.device, bsz, seq_len)
        h = self.tok_embeddings(tokens)
        for i, layer in enumerate(self.layers):
            h, layer_cache = layer(h, mask, pos, cache[i] if cache is not None else None)
            if cache is not None:
                cache[i] = layer_cache
        h = self.norm(h)
        if return_hidden:
            return h
        if targets is not None:
            if self.loss_fn is not None:
                loss = self.loss_fn(self.output(h).flatten(0, 1), targets.flatten(0, 1))
            else:
                loss = fused_matmul_cross_entropy(h, self.output.weight, targets)
            return weighted_mean_loss(loss, targets_mask=targets_mask)
        out = self.output(h)
        return (out, cache) if cache is not None else out

    def init_weights(self) -> None:
        args = self.args
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            torch.manual_seed(args.seed)
            std = args.init_base_std or args.dim ** -0.5
            nn.init.trunc_normal_(self.tok_embeddings.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
            self.norm.reset_parameters()
            for depth, block in enumerate(self.blocks()):
                factor = layer_init_factor(args.init_method, depth, self.n_layers, args.dim)
                block.init_weights(args.init_base_std, factor=factor)
            if not args.weight_tying:
                nn.init.trunc_normal_(self.output.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
