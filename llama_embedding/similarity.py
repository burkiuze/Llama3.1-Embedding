"""Similarity computations between embedding matrices.

Only two things live here:

1. :func:`cosine_similarity` — pairwise cosine similarity between two ``[N, D]``
   matrices (or the two rows of an ``[N, D]`` matrix against itself).
2. :func:`similarity_matrix` — the ``[B, B]`` query-document block used by the
   contrastive loss and by retrieval evaluation.

Cosine is the default metric everywhere in this project because the encoder
emits L2-normalised vectors, in which case cosine similarity and dot product
are numerically identical but cosine remains the semantically explicit choice
(and is still correct if a caller disables normalisation).
"""

from __future__ import annotations

import torch

from .projection import l2_normalize

__all__ = ["cosine_similarity", "dot_product", "euclidean_distance", "similarity_matrix"]


def cosine_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float = 1e-12,
    dim: int = -1,
) -> torch.Tensor:
    """Cosine similarity along ``dim``.

    Broadcasting follows ``torch`` conventions, so ``[N, D] x [N, D]`` with
    ``dim=-1`` yields ``[N]``, while ``[Q, D] x [C, D]`` requires an explicit
    unsqueeze on the left operand (see :func:`similarity_matrix`).
    """
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(f"feature dims differ: {a.shape[-1]} vs {b.shape[-1]}")
    a_norm = a.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)
    b_norm = b.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)
    return (a / a_norm * b / b_norm).sum(dim=dim)


def dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise dot product of two ``[N, D]`` matrices."""
    if a.shape != b.shape:
        raise ValueError(f"dot_product expects matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}")
    return (a * b).sum(dim=-1)


def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise euclidean distance between two ``[N, D]`` matrices."""
    if a.shape != b.shape:
        raise ValueError(
            f"euclidean_distance expects matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    return (a - b).norm(p=2, dim=-1)


def similarity_matrix(
    queries: torch.Tensor,
    documents: torch.Tensor,
    *,
    metric: str = "cosine",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Build the ``[num_queries, num_documents]`` similarity matrix.

    ``metric='cosine'`` (default) L2-normalises both sides first, which makes
    the result invariant to embedding magnitude and identical to a dot product
    on already-normalised vectors.
    """
    if queries.dim() != 2 or documents.dim() != 2:
        raise ValueError(
            f"queries/documents must be 2-D [N, D]; got {tuple(queries.shape)} and {tuple(documents.shape)}"
        )
    if queries.shape[-1] != documents.shape[-1]:
        raise ValueError(
            f"embedding dims differ: {queries.shape[-1]} vs {documents.shape[-1]}"
        )
    if metric == "cosine":
        q = l2_normalize(queries, eps)
        d = l2_normalize(documents, eps)
        return q @ d.T
    if metric == "dot":
        return queries @ documents.T
    if metric == "euclidean":
        # Negative distance: larger is better, consistent with similarity.
        return -torch.cdist(queries, documents, p=2)
    raise ValueError(f"unsupported metric {metric!r}")


def gather_inbatch_scores(
    queries: torch.Tensor,
    documents: torch.Tensor,
    *,
    metric: str = "cosine",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return the ``[B, B]`` in-batch block where row ``i`` scores ``document i`` positive.

    This is the matrix consumed by the InfoNCE objective.
    """
    return similarity_matrix(queries, documents, metric=metric, eps=eps)
