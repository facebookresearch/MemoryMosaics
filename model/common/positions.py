# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Where each token sits: its position, its document, the documents before it.

A model's create_attention_mask records the step's document layout in a DocumentLayout;
forward turns it into Positions, which every layer reads. Under "doc_causal" positions
count from the start of the current document, as in training, both at prefill and
while decoding -- including after a document boundary produced during decoding
(DocumentLayout.end_documents).
"""

from dataclasses import dataclass

import torch

# (keys, values, number of filled positions)
KVCache = tuple[torch.Tensor, torch.Tensor, int]


@dataclass
class Positions:
    """Per-token position inputs for one forward pass.

    tokenid: 1-based position of each token -- within its document under "doc_causal",
        within the row otherwise. Drives RoPE and the Mosaics short/long bandwidth.
    cluster_tokenid: number of tokens in earlier documents of the same row, i.e. the
        size of the cluster memory. None unless "doc_causal".
    doc_lens: document lengths, row-major, under "doc_causal" at a full-sequence step
        (training, evaluation, prefill). DeltaNet resets its state at each.
    doc_start: at a decoding step under "doc_causal", where each row's current document
        starts (bsz, 1); in-document memories must not read before it.
    """
    tokenid: torch.Tensor | None = None
    cluster_tokenid: torch.Tensor | None = None
    doc_lens: torch.Tensor | None = None
    doc_start: torch.Tensor | None = None


class DocumentLayout:
    """The document structure of the current step, as given to create_attention_mask."""

    def __init__(self) -> None:
        self.bias_type = "causal"
        self.q_lens: torch.Tensor | None = None   # document lengths (full-sequence step)
        self.start_pos: int | None = None         # position of the token (decoding step)
        self.n_rows = 0
        self.next_pos = 0                         # position after this step's tokens
        self.doc_start: torch.Tensor | None = None  # (bsz, 1), kept across decoding steps

    @property
    def documentwise(self) -> bool:
        return self.bias_type == "doc_causal"

    def reset(self) -> None:
        """Forget the carried document state (a new generation)."""
        self.doc_start = None

    def full_sequence(self, bias_type: str, q_lens: torch.Tensor, seq_len: int) -> None:
        self.bias_type, self.q_lens, self.start_pos = bias_type, q_lens, None
        self.next_pos = seq_len

    def decoding(self, bias_type: str, start_pos: int, n_rows: int) -> None:
        self.bias_type, self.q_lens, self.start_pos = bias_type, None, start_pos
        self.n_rows, self.next_pos = n_rows, start_pos + 1

    def end_documents(self, rows: torch.Tensor) -> None:
        """Rows flagged in `rows` (bool [bsz]) just read an end-of-document token: their
        next token starts a new document. No-op unless "doc_causal"."""
        if self.documentwise and self.doc_start is not None:
            self.doc_start = torch.where(rows.view(-1, 1), self.next_pos, self.doc_start)

    @torch.compiler.disable(recursive=True)
    def positions(self, device, bsz: int, seq_len: int) -> Positions:
        if self.q_lens is not None:
            if not self.documentwise:
                tokenid = torch.arange(1, seq_len + 1, dtype=torch.float32, device=device)
                return Positions(tokenid=tokenid.view(1, seq_len).repeat(bsz, 1))
            q_lens = self.q_lens
            doc_starts = torch.zeros(len(q_lens) + 1, dtype=torch.long, device=device)
            doc_starts[1:] = q_lens.cumsum(0)
            abs_start = doc_starts[:-1][torch.repeat_interleave(q_lens)]     # per token
            tokenid = torch.arange(doc_starts[-1], dtype=torch.float32, device=device) - abs_start + 1
            cluster_tokenid = (abs_start - (abs_start // seq_len) * seq_len).float()
            # Each row's last document continues into the decoding steps.
            last_col = torch.arange(1, bsz + 1, device=device) * seq_len - 1
            self.doc_start = cluster_tokenid[last_col].view(bsz, 1)
            return Positions(tokenid=tokenid, cluster_tokenid=cluster_tokenid, doc_lens=q_lens)
        if self.start_pos is None:
            raise RuntimeError("call create_attention_mask before forward")
        tokenid = torch.full((self.n_rows,), self.start_pos + 1.0, device=device)
        if not self.documentwise:
            return Positions(tokenid=tokenid)
        if self.doc_start is None:      # generation began without a prefill: at position 0
            self.doc_start = torch.zeros(self.n_rows, 1, device=device)
        return Positions(tokenid=tokenid - self.doc_start.view(-1), cluster_tokenid=self.doc_start,
                         doc_start=self.doc_start)
