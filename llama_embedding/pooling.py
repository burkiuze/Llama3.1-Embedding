"""Pooling strategies: token hidden states -> sentence representation.

Design
------
Pooling is intentionally isolated behind a single small abstraction so that no
other module in the codebase ever has to know *how* a sentence vector is
reduced from ``[B, T, H]`` token states. The rest of the pipeline only calls
``pooling(hidden_states, attention_mask)``.

Every strategy here is **mask aware**: ``attention_mask[b, t] == 0`` marks a
padding (or otherwise excluded) position and must never influence the returned
vector. Padding is therefore provably excluded rather than merely unlikely to
contribute.

Contract
--------
``forward(hidden_states, attention_mask)`` returns ``[B, H]`` and expects:

* ``hidden_states``: ``[B, T, H]`` float tensor.
* ``attention_mask``: ``[B, T]`` tensor of 1/0 (bool or int) or ``[B, T, T]``
  4-D mask (attention-style masks are collapsed to their per-position support).
* Rows whose mask is entirely zero are rejected with ``ValueError`` rather than
  silently producing a zero/NaN vector.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "PoolingStrategy",
    "MeanPooling",
    "LastTokenPooling",
    "WeightedMeanPooling",
    "get_pooling",
    "build_pooling",
]


def _prepare_mask(
    attention_mask: torch.Tensor, batch_size: int, seq_len: int, dtype: torch.dtype
) -> torch.Tensor:
    """Normalise any accepted mask layout into a float ``[B, T]`` support mask.

    Accepted layouts:
      * ``[B, T]``     — the standard HF ``attention_mask`` (1 = real token).
      * ``[B, T, T]`` / ``[B, 1, T, T]`` — attention-style masks. These are
        collapsed to the per-query-position support: a query position counts as
        real if it may attend to at least one key.
    """
    if attention_mask is None:
        raise ValueError("attention_mask is required for mask-aware pooling")
    if attention_mask.dim() == 4:
        if attention_mask.shape[-1] != seq_len:
            raise ValueError(
                f"4-D mask of shape {tuple(attention_mask.shape)} is not compatible "
                f"with seq_len={seq_len}"
            )
        attention_mask = attention_mask.amax(dim=-1)  # [B, Hq, T] -> collapse heads
    if attention_mask.dim() == 3:
        if attention_mask.shape[-1] != seq_len:
            raise ValueError(
                f"3-D mask of shape {tuple(attention_mask.shape)} is not compatible "
                f"with seq_len={seq_len}"
            )
        attention_mask = attention_mask.amax(dim=-1)
    if attention_mask.dim() != 2:
        raise ValueError(
            f"attention_mask must be [B, T], [B, T, T] or [B, 1, T, T]; "
            f"got {tuple(attention_mask.shape)}"
        )
    if attention_mask.shape[0] != batch_size or attention_mask.shape[1] != seq_len:
        raise ValueError(
            f"attention_mask shape {tuple(attention_mask.shape)} does not match "
            f"hidden states [B={batch_size}, T={seq_len}, ...]"
        )
    return attention_mask.to(dtype)


def _check_non_empty(mask: torch.Tensor) -> None:
    if bool((mask.sum(dim=-1) <= 0).any()):
        raise ValueError(
            "attention_mask contains a row with no active tokens; "
            "an all-padding input cannot produce a sentence vector"
        )


class PoolingStrategy(nn.Module):
    """Base class for all pooling strategies.

    Subclasses implement :meth:`pool`. The base class owns argument checking and
    the ``[B, T, H] -> [B, H]`` contract so the concrete strategies stay tiny.
    """

    def __init__(self) -> None:
        super().__init__()

    def pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        **kwargs: object,
    ) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must be [B, T, H]; got {tuple(hidden_states.shape)}"
            )
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        mask = _prepare_mask(attention_mask, batch_size, seq_len, hidden_states.dtype)
        _check_non_empty(mask)
        pooled = self.pool(hidden_states, mask, **kwargs)
        if pooled.shape != (batch_size, hidden_states.shape[2]):
            raise RuntimeError(
                f"{type(self).__name__} produced {tuple(pooled.shape)}, expected "
                f"{(batch_size, hidden_states.shape[2])}"
            )
        return pooled

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


class MeanPooling(PoolingStrategy):
    """Attention-mask-aware mean pooling (the default).

    ``out[b] = sum_t mask[b, t] * h[b, t] / sum_t mask[b, t]``

    Padding positions are multiplied out before summation, so they cannot
    influence the result even when the padding hidden states are arbitrary.
    """

    def pool(self, hidden_states: torch.Tensor, mask: torch.Tensor, **_: object) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


class LastTokenPooling(PoolingStrategy):
    """Representation of the final non-padding token of each sequence.

    The index is taken as ``argmax`` over the mask so that a right-padded batch
    (the tokenizer default) yields the correct final real token. Sequences that
    are left-padded or have interior gaps are resolved by using the largest
    index whose mask is active, which matches "last meaningful token" semantics.
    """

    def pool(self, hidden_states: torch.Tensor, mask: torch.Tensor, **_: object) -> torch.Tensor:
        seq_len = mask.shape[1]
        positions = torch.arange(seq_len, device=mask.device).unsqueeze(0)
        # Largest active index per row; rows are guaranteed non-empty by base.
        last_index = torch.where(
            mask > 0, positions.expand_as(mask), torch.full_like(positions.expand_as(mask), -1)
        ).amax(dim=-1)
        batch_index = torch.arange(mask.shape[0], device=mask.device)
        return hidden_states[batch_index, last_index]


class WeightedMeanPooling(PoolingStrategy):
    """Mask-aware mean pooling with optional per-token weights.

    ``out[b] = sum_t w[b, t] * mask[b, t] * h[b, t] / sum_t w[b, t] * mask[b, t]``

    ``weights`` is ``[B, T]`` and is normalised per row so only its *relative*
    distribution matters. This covers position-weighted schemes (e.g. decaying
    toward the end of the document) while keeping padding excluded.
    """

    def pool(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        **_: object,
    ) -> torch.Tensor:
        if weights is None:
            return MeanPooling().pool(hidden_states, mask)
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        if weights.dim() != 2 or weights.shape != mask.shape:
            raise ValueError(
                f"weights must be [B, T] matching the mask {tuple(mask.shape)}; "
                f"got {tuple(weights.shape)}"
            )
        weights = weights.to(dtype=hidden_states.dtype, device=hidden_states.device)
        # Guard against negative weights flipping the effective support.
        weights = weights.clamp(min=0.0)
        effective = weights * mask
        denom = effective.sum(dim=1, keepdim=True).clamp(min=1e-9)
        if bool((denom <= 1e-9).any()):
            raise ValueError("all weights are zero for at least one sequence")
        return (hidden_states * effective.unsqueeze(-1)).sum(dim=1) / denom


_POOLING_REGISTRY = {
    "mean": MeanPooling,
    "last_token": LastTokenPooling,
    "weighted_mean": WeightedMeanPooling,
}


def get_pooling(name: str) -> PoolingStrategy:
    """Instantiate a pooling strategy by name (raises ``KeyError`` if unknown)."""
    key = name.lower().strip()
    if key not in _POOLING_REGISTRY:
        raise KeyError(f"unknown pooling strategy {name!r}; expected one of {sorted(_POOLING_REGISTRY)}")
    return _POOLING_REGISTRY[key]()


def build_pooling(config: object) -> PoolingStrategy:
    """Build the pooling strategy named by an :class:`~llama_embedding.config.EmbeddingConfig`."""
    name = getattr(config, "pooling", None)
    if name is None:
        raise ValueError("config object has no 'pooling' attribute")
    return get_pooling(name)
