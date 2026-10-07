"""Batch collation: raw texts -> tokenised, padded tensors.

Separated from the dataset and the trainer so that tokenisation policy lives in
exactly one place: padding side, max length, truncation and prompt templates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from llama_embedding.config import PromptTemplates
from llama_embedding.tokenizer import tokenize_texts

__all__ = ["ContrastiveCollator", "encode_text_group"]


def encode_text_group(
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int,
    padding_side: str = "right",
    prompt: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Tokenize one group of texts into padded tensors.

    ``prompt`` is an optional prefix template (query/document) applied to every
    text in the group before tokenisation, so both sides of a pair always share
    the exact same treatment.
    """
    prepared: List[str] = [f"{prompt}{t}" if prompt else t for t in texts]
    previous_side = getattr(tokenizer, "padding_side", padding_side)
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = padding_side
    try:
        encoded = tokenize_texts(
            tokenizer, prepared, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
        )
    finally:
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = previous_side
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


@dataclass
class ContrastiveCollator:
    """Collate ``queries``/``positives``/``negatives`` into padded tensors.

    Handles the standard triplet layout where ``negatives`` is a list of
    ``[B, K]`` strings, flattening to ``[B * K]`` for one tokenisation pass.
    """

    tokenizer: Any
    max_length: int = 256
    padding_side: str = "right"
    prompts: Optional[PromptTemplates] = None

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if self.tokenizer is None:
            raise ValueError("a tokenizer is required")

    def _template(self, role: str) -> Optional[str]:
        if self.prompts is None:
            return None
        return getattr(self.prompts, role, None) or None

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty batch")
        if "queries" not in features[0] or "positives" not in features[0]:
            raise ValueError("each feature must provide 'queries' and 'positives'")

        queries: List[str] = [f["queries"] for f in features]
        positives: List[str] = [f["positives"] for f in features]
        negatives_per_item: List[List[str]] = [list(f.get("negatives", [])) for f in features]

        num_negatives = max((len(n) for n in negatives_per_item), default=0)
        if num_negatives > 0:
            # The tensor layout is a dense [B, K, T], so every item needs the same
            # K. Items that supplied fewer negatives are padded by repeating one
            # of their own negatives (the max-over-negatives objective is
            # unaffected by a repeat). Padding with "" would be a bug: it would
            # tokenise to a real BOS/empty row and inject a meaningless vector
            # into the hard-negative loss.
            for i, negatives in enumerate(negatives_per_item):
                while len(negatives) < num_negatives:
                    fallback = negatives[0] if negatives else negatives_per_item[0][0]
                    negatives.append(fallback)
                if not negatives:  # no negatives anywhere in the batch
                    negatives_per_item[i] = []
                    break

        flat_negatives: List[str] = []
        for negatives in negatives_per_item:
            flat_negatives.extend(negatives)

        batch: Dict[str, Any] = {}
        batch["query"] = encode_text_group(
            self.tokenizer, queries, max_length=self.max_length,
            padding_side=self.padding_side, prompt=self._template("query"),
        )
        batch["positive"] = encode_text_group(
            self.tokenizer, positives, max_length=self.max_length,
            padding_side=self.padding_side, prompt=self._template("document"),
        )

        if num_negatives > 0:
            negative_encoded = encode_text_group(
                self.tokenizer, flat_negatives, max_length=self.max_length,
                padding_side=self.padding_side, prompt=self._template("document"),
            )
            batch_size = len(queries)
            # flat_negatives is item-major: [b0_n0, b0_n1, ..., b1_n0, b1_n1, ...]
            # so the encoded [B*K, T] tensor reshapes directly to [B, K, T].
            seq_len = negative_encoded["input_ids"].shape[-1]
            batch["negative"] = {
                "input_ids": negative_encoded["input_ids"].view(batch_size, num_negatives, seq_len),
                "attention_mask": negative_encoded["attention_mask"].view(
                    batch_size, num_negatives, seq_len
                ),
            }
            batch["num_negatives"] = num_negatives
        else:
            batch["num_negatives"] = 0

        batch["size"] = len(queries)
        return batch


@dataclass
class EvalCollator:
    """Simpler collator for plain lists of texts (used by evaluation)."""

    tokenizer: Any
    max_length: int = 512

    def __call__(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        return encode_text_group(self.tokenizer, list(texts), max_length=self.max_length)
