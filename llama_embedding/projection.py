"""Embedding projection head: pooled hidden states -> normalised dense vectors.

The head is deliberately small and independent from the Llama backbone:

    pooled [B, hidden_size]
        -> Linear(hidden_size -> embedding_dim)
        -> (optional) LayerNorm
        -> (optional) L2 normalisation

Why a projection at all?
-----------------------
Raw Llama hidden states live in ``R^4096`` with an arbitrary scale and are
tuned for next-token prediction, not for sentence-level cosine geometry. A
learned linear map plus normalisation gives the contrastive loss a well
conditioned, fixed-size target and decouples the embedding dimensionality from
the backbone width, which is what makes the 256/384/512/768/1024 options and
future Matryoshka-style truncation possible.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SUPPORTED_EMBEDDING_DIMS

__all__ = ["EmbeddingHead", "l2_normalize", "get_activation"]


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise L2 normalisation.

    Uses ``F.normalize`` semantics with an explicit epsilon so that zero vectors
    return zeros instead of NaNs. Rows therefore satisfy ``||x||_2 ~= 1`` for any
    non-zero input, which is what the cosine-similarity path assumes.
    """
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


def get_activation(name: Optional[str]) -> nn.Module:
    """Return a small activation module by name (``None`` for identity)."""
    if name is None or name == "":
        return nn.Identity()
    key = name.lower()
    if key in ("identity", "none", "linear"):
        return nn.Identity()
    if key == "gelu":
        return nn.GELU()
    if key == "relu":
        return nn.ReLU()
    if key in ("silu", "swish"):
        return nn.SiLU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError(f"unsupported activation {name!r}")


class EmbeddingHead(nn.Module):
    """Small trainable projection from backbone hidden size to embedding size.

    Parameters
    ----------
    hidden_size:
        Backbone hidden width (Llama 3.1 8B = 4096).
    embedding_dim:
        Output dimensionality; validated against ``SUPPORTED_EMBEDDING_DIMS``.
    use_layer_norm:
        Insert ``LayerNorm`` after the linear map (configurable).
    normalize:
        L2-normalise the output. When ``True`` (the default) every returned row
        has unit L2 norm.
    activation:
        Optional non-linearity between the linear map and the LayerNorm. The
        default keeps the head a pure linear map so that Matryoshka prefix
        truncation remains meaningful.
    matryoshka_dims:
        Prefix dimensions the head is expected to be evaluated at. Configuration
        only: :meth:`truncate` slices an already-computed vector.
    """

    def __init__(
        self,
        hidden_size: int,
        embedding_dim: int = 768,
        *,
        use_layer_norm: bool = True,
        normalize: bool = True,
        bias: bool = False,
        activation: Optional[str] = None,
        layer_norm_eps: float = 1e-12,
        matryoshka_dims: Sequence[int] = (),
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if embedding_dim not in SUPPORTED_EMBEDDING_DIMS:
            raise ValueError(
                f"embedding_dim must be one of {SUPPORTED_EMBEDDING_DIMS}, got {embedding_dim}"
            )
        if not use_layer_norm and not normalize:
            # Not fatal, but almost always a configuration mistake: without
            # either operation the cosine geometry is unconstrained.
            import warnings

            warnings.warn(
                "EmbeddingHead has neither LayerNorm nor L2 normalisation; "
                "cosine similarity will still work but scale is unconstrained.",
                stacklevel=2,
            )

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        self.use_layer_norm = use_layer_norm

        self.linear = nn.Linear(hidden_size, embedding_dim, bias=bias)
        self.activation = get_activation(activation)
        self.layer_norm = (
            nn.LayerNorm(embedding_dim, eps=layer_norm_eps) if use_layer_norm else nn.Identity()
        )
        self.matryoshka_dims = tuple(matryoshka_dims or ())

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Project pooled states to embeddings.

        Parameters
        ----------
        pooled: ``[B, hidden_size]`` pooled sentence representations.

        Returns
        -------
        ``[B, embedding_dim]`` embeddings (unit norm when ``normalize=True``).
        """
        if pooled.dim() != 2:
            raise ValueError(f"expected pooled states [B, hidden_size]; got {tuple(pooled.shape)}")
        if pooled.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected pooled states with last dim {self.hidden_size}; got {pooled.shape[-1]}"
            )
        projected = self.linear(pooled)
        projected = self.activation(projected)
        projected = self.layer_norm(projected)
        if self.normalize:
            projected = l2_normalize(projected, eps=self.layer_norm.eps if self.use_layer_norm else 1e-12)
        return projected

    def truncate(self, embeddings: torch.Tensor, dim: int) -> torch.Tensor:
        """Return the leading ``dim`` coordinates of ``embeddings``, re-normalised.

        This is the Matryoshka-style read path. Truncation must be followed by a
        renormalisation to keep ``||v||_2 == 1``; the helper does both and
        validates that ``dim`` is one of the configured/known dimensions.
        """
        if dim not in SUPPORTED_EMBEDDING_DIMS:
            raise ValueError(f"unsupported truncation dim {dim}; expected {SUPPORTED_EMBEDDING_DIMS}")
        if dim > embeddings.shape[-1]:
            raise ValueError(
                f"cannot truncate to {dim} dims from a vector of width {embeddings.shape[-1]}"
            )
        sliced = embeddings[..., :dim]
        return l2_normalize(sliced)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"hidden_size={self.hidden_size}, embedding_dim={self.embedding_dim}, "
            f"layer_norm={self.use_layer_norm}, normalize={self.normalize}"
        )


def count_trainable(module: nn.Module) -> tuple[int, int]:
    """Return ``(trainable_parameters, total_parameters)`` for a module."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def freeze(module: nn.Module) -> None:
    """Freeze every parameter of ``module`` in place."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze(module: nn.Module) -> None:
    """Unfreeze every parameter of ``module`` in place."""
    for param in module.parameters():
        param.requires_grad = True
