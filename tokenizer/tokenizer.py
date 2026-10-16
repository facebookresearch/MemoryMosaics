# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Tokenizer: a HuggingFace SentencePiece tokenizer, Mistral-7B's by default.

The models in this repository are trained with the 32k byte-fallback SentencePiece
vocabulary of mistralai/Mistral-7B-v0.1 (the same as Mixtral-8x7B-v0.1). The repository
is gated on HuggingFace: `huggingface-cli login` first, or pass a local directory.
Special tokens are added by the caller (encode's bos/eos flags), never implicitly.
"""

import logging

logger = logging.getLogger()

DEFAULT_TOKENIZER = "mistralai/Mistral-7B-v0.1"


class Tokenizer:
    def __init__(self, path: str | None = None) -> None:
        """`path`: a HuggingFace repo id or a local directory (None: Mistral-7B)."""
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError("the tokenizer needs transformers and sentencepiece: "
                              "pip install transformers sentencepiece") from e
        self.tokenizer_path = path or DEFAULT_TOKENIZER
        self._hf = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self._bos_id = self._hf.bos_token_id
        self._eos_id = self._hf.eos_token_id
        # Mistral defines no pad token. <unk> never occurs in byte-fallback text, so it
        # pads without being mistaken for anything -- in particular not for EOS, which
        # marks document boundaries.
        pad = self._hf.pad_token_id
        self._pad_id = pad if pad is not None else self._hf.unk_token_id
        desc = (f"{self.tokenizer_path} ({type(self._hf).__name__}, {self.n_words} tokens, "
                f"bos={self._bos_id} eos={self._eos_id} pad={self._pad_id})")
        assert None not in (self._bos_id, self._eos_id, self._pad_id), \
            f"{desc}: needs bos, eos and a pad (or unk) token"
        # Pad marks no document boundary (EOS does) and is masked from the loss.
        assert len({self._bos_id, self._eos_id, self._pad_id}) == 3, \
            f"{desc}: bos, eos and pad must be three different tokens"
        logger.info(f"Tokenizer {desc}")

    @property
    def n_words(self) -> int:
        return len(self._hf)

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def pad_id(self) -> int:
        return self._pad_id

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        tokens = self._hf.encode(text, add_special_tokens=False)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: list[int] | int) -> str:
        return self._hf.decode([tokens] if isinstance(tokens, int) else tokens)
