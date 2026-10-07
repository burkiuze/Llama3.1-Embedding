"""Retrieval metrics: Recall@K, MRR, NDCG@K.

All functions operate on a **ranking matrix** and a **relevance matrix** of the
same shape, which keeps them independent of any encoder and trivially testable.

Conventions
-----------
* ``rankings[i]`` holds document indices ordered best-first for query ``i``.
* ``relevances[i]`` holds per-document relevance scores aligned with the
  *original* document order, where ``> 0`` means "relevant".
* Metrics are macro-averaged over queries.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

__all__ = [
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "hit_rate_at_k",
    "mean_average_precision",
    "retrieval_metrics",
    "ranking_from_scores",
    "evaluate_rankings",
]


def _validate(rankings: torch.Tensor, relevances: torch.Tensor) -> None:
    if rankings.shape != relevances.shape:
        raise ValueError(
            f"rankings {tuple(rankings.shape)} and relevances {tuple(relevances.shape)} must match"
        )
    if rankings.dim() != 2:
        raise ValueError(f"expected 2-D [num_queries, k]; got {tuple(rankings.shape)}")


def ranking_from_scores(scores: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
    """Return document indices sorted by descending score, per query row."""
    if scores.dim() != 2:
        raise ValueError(f"scores must be [num_queries, num_documents]; got {tuple(scores.shape)}")
    k = scores.shape[1] if top_k is None else min(top_k, scores.shape[1])
    _, indices = torch.topk(scores, k=k, dim=-1)
    return indices


def recall_at_k(rankings: torch.Tensor, relevances: torch.Tensor, k: int = 10) -> float:
    """Recall@K: fraction of relevant documents retrieved in the top ``k``."""
    _validate(rankings, relevances)
    k = min(k, rankings.shape[1])
    if k <= 0:
        raise ValueError("k must be positive")
    top = rankings[:, :k]
    rel_at_k = relevances.gather(1, top)
    num_relevant = (relevances > 0).sum(dim=1).clamp(min=1)
    hits = (rel_at_k > 0).sum(dim=1).float()
    return float((hits / num_relevant).mean())


def precision_at_k(rankings: torch.Tensor, relevances: torch.Tensor, k: int = 10) -> float:
    """Precision@K: fraction of the top ``k`` results that are relevant."""
    _validate(rankings, relevances)
    k = min(k, rankings.shape[1])
    if k <= 0:
        raise ValueError("k must be positive")
    rel_at_k = relevances.gather(1, rankings[:, :k])
    return float((rel_at_k > 0).float().mean())


def hit_rate_at_k(rankings: torch.Tensor, relevances: torch.Tensor, k: int = 10) -> float:
    """1.0 if any relevant document appears in the top ``k``, averaged over queries."""
    _validate(rankings, relevances)
    k = min(k, rankings.shape[1])
    rel_at_k = relevances.gather(1, rankings[:, :k])
    return float((rel_at_k > 0).any(dim=1).float().mean())


def mrr(rankings: torch.Tensor, relevances: torch.Tensor) -> float:
    """Mean Reciprocal Rank over the full ranking depth.

    Queries with no relevant document in the ranking are skipped (the standard
    behaviour); if that leaves nothing, the result is 0.0.
    """
    _validate(rankings, relevances)
    rel = relevances.gather(1, rankings)  # relevance in rank order
    is_rel = (rel > 0).float()
    positions = torch.arange(1, rankings.shape[1] + 1, device=rankings.device, dtype=torch.float32)
    reciprocal = is_rel / positions.unsqueeze(0)
    has_any = is_rel.sum(dim=1) > 0
    if not bool(has_any.any()):
        return 0.0
    return float(reciprocal.sum(dim=1)[has_any].mean())


def dcg_at_k(relevances_in_rank_order: torch.Tensor, k: int) -> torch.Tensor:
    positions = torch.arange(1, k + 1, device=relevances_in_rank_order.device, dtype=torch.float32)
    gains = (2.0**relevances_in_rank_order[:, :k] - 1.0) / torch.log2(positions.unsqueeze(0) + 1.0)
    return gains.sum(dim=1)


def ndcg_at_k(rankings: torch.Tensor, relevances: torch.Tensor, k: int = 10) -> float:
    """NDCG@K with exponential gain."""
    _validate(rankings, relevances)
    k = min(k, rankings.shape[1])
    rel_in_rank = relevances.gather(1, rankings[:, :k])

    ideal, _ = torch.sort(relevances, dim=1, descending=True)
    ideal_top = ideal[:, :k]

    dcg = dcg_at_k(rel_in_rank, k)
    idcg = dcg_at_k(ideal_top, k)
    ndcg = torch.where(idcg > 0, dcg / idcg.clamp(min=1e-12), torch.zeros_like(dcg))
    # Queries with no relevant documents contribute 0 rather than a 0/0 NaN.
    return float(ndcg.mean())


def mean_average_precision(rankings: torch.Tensor, relevances: torch.Tensor) -> float:
    """MAP over the full ranking depth."""
    _validate(rankings, relevances)
    rel = (relevances.gather(1, rankings) > 0).float()
    total_relevant = (relevances > 0).sum(dim=1).clamp(min=1)
    precision_at_hit = torch.cumsum(rel, dim=1) / torch.arange(
        1, rankings.shape[1] + 1, device=rankings.device, dtype=torch.float32
    ).unsqueeze(0)
    ap = (precision_at_hit * rel).sum(dim=1) / total_relevant
    has_rel = (relevances > 0).sum(dim=1) > 0
    if not bool(has_rel.any()):
        return 0.0
    return float(ap[has_rel].mean())


def retrieval_metrics(
    rankings: torch.Tensor,
    relevances: torch.Tensor,
    *,
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Compute the standard retrieval metric bundle."""
    _validate(rankings, relevances)
    results: Dict[str, float] = {}
    for k in ks:
        effective_k = min(k, rankings.shape[1])
        results[f"recall@{k}"] = recall_at_k(rankings, relevances, effective_k)
        results[f"precision@{k}"] = precision_at_k(rankings, relevances, effective_k)
        results[f"hit_rate@{k}"] = hit_rate_at_k(rankings, relevances, effective_k)
        results[f"ndcg@{k}"] = ndcg_at_k(rankings, relevances, effective_k)
    results["mrr"] = mrr(rankings, relevances)
    results["map"] = mean_average_precision(rankings, relevances)
    return results


def evaluate_rankings(
    scores: torch.Tensor,
    relevances: torch.Tensor,
    *,
    top_k: Optional[int] = None,
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Rank ``scores`` then compute metrics in one call."""
    rankings = ranking_from_scores(scores, top_k=top_k)
    depth = top_k or max(ks)
    rankings = rankings[:, : min(depth, rankings.shape[1])]
    return retrieval_metrics(rankings, relevances, ks=ks)
