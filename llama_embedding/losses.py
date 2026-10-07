"""Contrastive objectives for training sentence embeddings.

Primary objective
-----------------
:func:`info_nce` implements InfoNCE (a.k.a. in-batch softmax cross-entropy)
over an ``[B, B]`` similarity block where the diagonal holds the
query/positive pairs and every off-diagonal entry is an in-batch negative.

    L = -1/B * sum_i log( exp(sim(q_i, d_i)/tau) / sum_j exp(sim(q_i, d_j)/tau) )

Temperature
-----------
``temperature`` (``tau``) scales the logits. The software default is
``DEFAULT_TEMPERATURE`` (0.02) — the value used as a default by widely deployed
sentence-embedding implementations. **It is a documented default, not a tuned
optimum.** Sweep it on a validation split; both the loss and the
retrieval metrics are sensitive to it.

Hard negatives
--------------
Explicitly provided negatives can be scored alongside in-batch negatives via
:func:`pairwise_info_nce`, which is the ``(query, positive, hard_negative)``
triplet objective. It is composable with the in-batch term and is enabled when
the dataset supplies negatives.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import DEFAULT_TEMPERATURE
from .projection import l2_normalize
from .similarity import similarity_matrix

__all__ = [
    "info_nce",
    "pairwise_info_nce",
    "matryoshka_info_nce",
    "contrastive_loss",
    "build_matryoshka_prefixes",
    "DEFAULT_TEMPERATURE",
]


def _prepare(
    queries: torch.Tensor,
    documents: torch.Tensor,
    *,
    normalize: bool,
    metric: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if queries.dim() != 2 or documents.dim() != 2:
        raise ValueError(
            f"embeddings must be 2-D [B, D]; got {tuple(queries.shape)} and {tuple(documents.shape)}"
        )
    if queries.shape[0] != documents.shape[0]:
        raise ValueError(
            f"batch size mismatch: {queries.shape[0]} queries vs {documents.shape[0]} documents"
        )
    if queries.shape[1] != documents.shape[1]:
        raise ValueError(
            f"embedding dim mismatch: {queries.shape[1]} vs {documents.shape[1]}"
        )
    if queries.shape[0] < 2:
        raise ValueError(
            "in-batch contrastive training needs a batch size of at least 2 "
            "(with B=1 there are no in-batch negatives)"
        )
    if normalize:
        queries = l2_normalize(queries)
        documents = l2_normalize(documents)
    return queries, documents


def info_nce(
    queries: torch.Tensor,
    documents: torch.Tensor,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    metric: str = "cosine",
    normalize: bool = True,
    symmetric: bool = False,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """In-batch InfoNCE / MultipleNegativesRanking loss.

    Parameters
    ----------
    queries, documents:
        ``[B, D]`` embeddings. ``documents[i]`` is the positive of ``queries[i]``.
    temperature:
        Logit scale ``1/tau``. Tunable; see module docstring.
    metric:
        ``cosine`` (default), ``dot`` or ``euclidean``.
    normalize:
        Re-normalise inputs even if the encoder already did (defensive, cheap).
    symmetric:
        Also average the document-to-query direction.
    label_smoothing:
        Forward KL smoothing towards a uniform distribution over the batch.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")

    q, d = _prepare(queries, documents, normalize=normalize, metric=metric)
    logits = similarity_matrix(q, d, metric=metric) / temperature
    target = torch.arange(logits.shape[0], device=logits.device)
    loss = F.cross_entropy(logits, target, label_smoothing=label_smoothing)
    if symmetric:
        loss = 0.5 * (loss + F.cross_entropy(logits.T, target, label_smoothing=label_smoothing))
    return loss


def pairwise_info_nce(
    queries: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    metric: str = "cosine",
    normalize: bool = True,
) -> torch.Tensor:
    """Triplet contrastive loss over ``(query, positive, negative)``.

    ``negatives`` may be ``[B, D]`` (one hard negative per query) or ``[B, K, D]``
    (K hard negatives per query). Uses a max-over-negatives formulation so the
    hardest supplied negative dominates the gradient.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if queries.shape[0] != positives.shape[0]:
        raise ValueError("queries and positives must have the same batch size")
    if negatives.dim() == 2:
        negatives = negatives.unsqueeze(1)
    if negatives.dim() != 3:
        raise ValueError(f"negatives must be [B, D] or [B, K, D]; got {tuple(negatives.shape)}")
    if negatives.shape[0] != queries.shape[0]:
        raise ValueError("negatives batch size must match queries")

    if normalize:
        q = l2_normalize(queries)
        p = l2_normalize(positives)
        n = l2_normalize(negatives)
    else:
        q, p, n = queries, positives, negatives

    pos_scores = (q * p).sum(dim=-1).unsqueeze(1) if metric == "cosine" else (q * p).sum(-1, keepdim=True)
    if metric == "cosine":
        neg_scores = torch.einsum("bd,bkd->bk", q, n)
    elif metric == "dot":
        neg_scores = torch.einsum("bd,bkd->bk", q, n)
    elif metric == "euclidean":
        pos_scores = -(q - p).norm(dim=-1, keepdim=True)
        neg_scores = -torch.cdist(q.unsqueeze(1), n).squeeze(1)
    else:
        raise ValueError(f"unsupported metric {metric!r}")

    logits = torch.cat([pos_scores, neg_scores], dim=1) / temperature
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


def build_matryoshka_prefixes(embedding_dim: int, dims: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return the Matryoshka loss dimensions (sorted, unique, capped by dim).

    Includes ``embedding_dim`` itself as the final (full) term.
    """
    valid = sorted({int(d) for d in dims if 0 < int(d) <= embedding_dim})
    if embedding_dim not in valid:
        valid.append(embedding_dim)
    return tuple(sorted(set(valid)))


def matryoshka_info_nce(
    queries: torch.Tensor,
    documents: torch.Tensor,
    dims: Tuple[int, ...],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    metric: str = "cosine",
    normalize: bool = True,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """InfoNCE applied to every Matryoshka prefix, summed over prefixes.

    Returns ``(loss, used_dims)``. The caller decides the weighting; the
    training loop averages uniformly unless a weight is configured.
    """
    used = build_matryoshka_prefixes(queries.shape[-1], tuple(dims))
    total: Optional[torch.Tensor] = None
    for d in used:
        loss_d = info_nce(
            queries[..., :d],
            documents[..., :d],
            temperature=temperature,
            metric=metric,
            normalize=normalize,
        )
        total = loss_d if total is None else total + loss_d
    assert total is not None
    return total / len(used), used


def contrastive_loss(
    queries: torch.Tensor,
    documents: torch.Tensor,
    *,
    negatives: Optional[torch.Tensor] = None,
    matryoshka_dims: Tuple[int, ...] = (),
    temperature: float = DEFAULT_TEMPERATURE,
    metric: str = "cosine",
    normalize: bool = True,
    symmetric: bool = False,
    label_smoothing: float = 0.0,
    hard_negative_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Compose the full contrastive objective used by the trainer.

    Combines (optionally) the in-batch InfoNCE term, an explicit hard-negative
    triplet term, and Matryoshka prefix terms, and returns a dict of named
    scalar losses so the trainer can log each component separately.

    ``hard_negative_weight`` blends the triplet term with the in-batch term
    (``0`` = in-batch only). Without explicit negatives the in-batch negatives
    already act as hard negatives after the first epoch.
    """
    metrics: Dict[str, torch.Tensor] = {}
    if matryoshka_dims:
        loss, used_dims = matryoshka_info_nce(
            queries,
            documents,
            tuple(matryoshka_dims),
            temperature=temperature,
            metric=metric,
            normalize=normalize,
        )
        metrics["matryoshka"] = loss
        metrics["matryoshka_dims"] = torch.tensor(float(len(used_dims)))
        total = loss
    else:
        in_batch = info_nce(
            queries,
            documents,
            temperature=temperature,
            metric=metric,
            normalize=normalize,
            symmetric=symmetric,
            label_smoothing=label_smoothing,
        )
        metrics["in_batch"] = in_batch
        total = in_batch

    if negatives is not None and hard_negative_weight > 0.0:
        hard = pairwise_info_nce(
            queries,
            documents,
            negatives,
            temperature=temperature,
            metric=metric,
            normalize=normalize,
        )
        metrics["hard_negative"] = hard
        total = (1.0 - hard_negative_weight) * total + hard_negative_weight * hard

    metrics["loss"] = total
    return metrics
