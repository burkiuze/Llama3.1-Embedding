"""Tokenizer helpers for the Llama 3.1 base backbone.

The tokenizer is used **as shipped**. Nothing here adds special tokens, changes
the vocabulary or alters the sentencepiece behaviour: tokenising is a solved
problem that the base model was pretrained with, and mutating it would only
break alignment between pretrained embeddings and our token ids.

This module exists to (a) centralise the padding/truncation conventions, (b)
hide the HF tokenizer behind a tiny stable interface so the rest of the code is
testable without transformers, and (c) support CPU-safe smoke tests.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

__all__ = ["TokenizedBatch", "build_tokenizer", "tokenize_texts", "DEFAULT_PAD_TOKEN_ID"]

#: Llama 3.1's tokenizer has no pad token; `<|end_of_text|>` (128001) is the
#: conventional padding id for the base tokenizer and is what we use by default.
DEFAULT_PAD_TOKEN_ID = 128001


class TokenizedBatch(dict):
    """A ``dict`` with attribute access over the standard tokenizer outputs."""

    @property
    def input_ids(self) -> List[List[int]]:
        return self["input_ids"]

    @property
    def attention_mask(self) -> List[List[int]]:
        return self["attention_mask"]

    @property
    def tokens(self) -> Optional[List[List[str]]]:
        return self.get("tokens")


def build_tokenizer(
    model_name_or_path: str,
    *,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
    padding_side: str = "right",
):
    """Load the backbone tokenizer.

    ``token`` may be supplied via the argument, the ``HF_TOKEN`` /
    ``HUGGINGFACE_TOKEN`` environment variables, or a local cache populated by
    ``huggingface-cli login``. Access to the official Llama 3.1 weights is gated:
    request access at https://www.llama.com/llama3_1/ and accept the licence
    before running this. The project never ships or bypasses those credentials.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "transformers is required to load the Llama tokenizer: pip install 'transformers>=4.43'"
        ) from exc

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    kwargs: Dict[str, object] = {
        "trust_remote_code": trust_remote_code,
        "use_fast": True,
    }
    if token:
        kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|end_of_text|>"
    tokenizer.padding_side = padding_side
    return tokenizer


def tokenize_texts(
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int = 512,
    padding: bool = True,
    truncation: bool = True,
    return_tensors: Optional[str] = None,
) -> TokenizedBatch:
    """Tokenize ``texts`` with the project's padding/truncation conventions.

    Right padding plus an explicit ``attention_mask`` is what makes the
    mask-aware pooling correct: trailing pad positions are masked out and can
    never enter the sentence vector.
    """
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    encoded = tokenizer(
        list(texts),
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors,
    )
    return TokenizedBatch(encoded)


class DummyTokenizer:
    """Whitespace tokenizer used by CPU-safe tests and smoke runs.

    It implements just enough of the HF tokenizer surface for
    :func:`tokenize_texts` and for tests that must never touch the network.
    ``vocab_size`` and ``pad_token_id`` mirror Llama 3.1 so shape assertions in
    tests resemble the real pipeline.
    """

    def __init__(self, vocab_size: int = 128256, pad_token_id: int = DEFAULT_PAD_TOKEN_ID) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.pad_token = "<pad>"
        self.eos_token = "<|end_of_text|>"
        self.padding_side = "right"

    def __call__(
        self,
        texts: Sequence[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: Optional[str] = None,
        **_: object,
    ) -> Dict[str, object]:
        batch_ids: List[List[int]] = []
        for text in texts:
            # Deterministic pseudo-ids: stable hash of each word.
            ids: List[int] = []
            for word in str(text).split():
                token_id = 1000 + (hash_word(word) % 1000)
                ids.append(token_id)
                if truncation and len(ids) >= max_length:
                    break
            batch_ids.append(ids or [self.pad_token_id])

        seq_lens = [len(ids) for ids in batch_ids]
        width = max(seq_lens) if padding else max(seq_lens)
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        for ids in batch_ids:
            pad = width - len(ids)
            if self.padding_side == "left":
                input_ids.append([self.pad_token_id] * pad + ids)
                attention_mask.append([0] * pad + [1] * len(ids))
            else:
                input_ids.append(ids + [self.pad_token_id] * pad)
                attention_mask.append([1] * len(ids) + [0] * pad)

        result: Dict[str, object] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_tensors == "pt":
            import torch

            result["input_ids"] = torch.tensor(input_ids, dtype=torch.long)
            result["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
        return result

    def decode(self, ids: Sequence[int], **_: object) -> str:  # pragma: no cover - parity helper
        return " ".join(str(int(i)) for i in ids)


def hash_word(word: str) -> int:
    """Deterministic (process-independent) string hash.

    Python's builtin ``hash`` is randomised per process, which would break the
    determinism guarantees the tests rely on, so a stable FNV-1a is used.
    """
    h = 0x811C9DC5
    for byte in word.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h
