"""Evaluation metric tests: Recall@K, MRR, NDCG@K and friends."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from evaluation.metrics import (
    evaluate_rankings,
    hit_rate_at_k,
    mean_average_precision,
    mrr,
    ndcg_at_k,
    precision_at_k,
    ranking_from_scores,
    recall_at_k,
    retrieval_metrics,
)


@pytest.fixture
def perfect_setup():
    """3 queries; the relevant document is ranked first in every case."""
    rankings = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    relevances = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return rankings, relevances


@pytest.fixture
def worst_setup():
    """3 queries; the relevant document is ranked last in every case."""
    rankings = torch.tensor([[2, 1, 0], [0, 2, 1], [1, 0, 2]])
    relevances = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return rankings, relevances


# --------------------------------------------------------------------------- #
# Recall@K
# --------------------------------------------------------------------------- #


def test_recall_perfect(perfect_setup):
    rankings, relevances = perfect_setup
    assert recall_at_k(rankings, relevances, k=1) == pytest.approx(1.0)
    assert recall_at_k(rankings, relevances, k=3) == pytest.approx(1.0)


def test_recall_worst_case_is_zero(worst_setup):
    rankings, relevances = worst_setup
    assert recall_at_k(rankings, relevances, k=1) == pytest.approx(0.0)


def test_recall_partial():
    """Two relevant docs, one retrieved at k=1."""
    rankings = torch.tensor([[0, 1, 2]])
    relevances = torch.tensor([[1.0, 1.0, 0.0]])
    assert recall_at_k(rankings, relevances, k=1) == pytest.approx(0.5)
    assert recall_at_k(rankings, relevances, k=2) == pytest.approx(1.0)


def test_recall_all_relevant_found():
    rankings = torch.tensor([[1, 2, 0]])
    relevances = torch.tensor([[1.0, 1.0, 1.0]])
    assert recall_at_k(rankings, relevances, k=3) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Precision / hit rate
# --------------------------------------------------------------------------- #


def test_precision_at_k():
    rankings = torch.tensor([[0, 1, 2]])
    relevances = torch.tensor([[1.0, 0.0, 0.0]])
    assert precision_at_k(rankings, relevances, k=1) == pytest.approx(1.0)
    assert precision_at_k(rankings, relevances, k=3) == pytest.approx(1 / 3)


def test_hit_rate_at_k():
    rankings = torch.tensor([[0, 1, 2], [0, 1, 2]])
    relevances = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert hit_rate_at_k(rankings, relevances, k=1) == pytest.approx(0.5)
    assert hit_rate_at_k(rankings, relevances, k=3) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# MRR
# --------------------------------------------------------------------------- #


def test_mrr_perfect(perfect_setup):
    rankings, relevances = perfect_setup
    assert mrr(rankings, relevances) == pytest.approx(1.0)


def test_mrr_worst(worst_setup):
    rankings, relevances = worst_setup
    assert mrr(rankings, relevances) == pytest.approx(1 / 3)


def test_mrr_second_position():
    rankings = torch.tensor([[1, 0]])
    relevances = torch.tensor([[1.0, 0.0]])
    assert mrr(rankings, relevances) == pytest.approx(0.5)


def test_mrr_skips_queries_without_relevant(perfect_setup):
    rankings, relevances = perfect_setup
    relevances_norel = torch.zeros_like(relevances)
    relevances_norel[0] = torch.tensor([1.0, 0.0, 0.0])
    # Only query 0 has a relevant doc; MRR should average over that one only.
    assert mrr(rankings, relevances_norel) == pytest.approx(1.0)


def test_mrr_no_relevant_returns_zero(perfect_setup):
    rankings, _ = perfect_setup
    assert mrr(rankings, torch.zeros(3, 3)) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# NDCG@K
# --------------------------------------------------------------------------- #


def test_ndcg_perfect(perfect_setup):
    rankings, relevances = perfect_setup
    assert ndcg_at_k(rankings, relevances, k=3) == pytest.approx(1.0)


def test_ndcg_worst_case_low(worst_setup):
    rankings, relevances = worst_setup
    assert ndcg_at_k(rankings, relevances, k=3) < 0.6


def test_ndcg_rewards_early_relevant():
    """Moving the relevant document earlier must increase NDCG."""
    relevances = torch.tensor([[1.0, 1.0, 1.0]])
    early = torch.tensor([[0, 1, 2]])
    late = torch.tensor([[2, 1, 0]])
    assert ndcg_at_k(early, relevances, k=3) > ndcg_at_k(late, relevances, k=3)


def test_ndcg_increases_with_k(perfect_setup):
    rankings, relevances = perfect_setup
    scores = [ndcg_at_k(rankings, relevances, k=k) for k in (1, 2, 3)]
    assert scores[0] <= scores[1] <= scores[2]


def test_ndcg_single_relevant_at_rank2():
    rankings = torch.tensor([[1, 0, 2]])
    relevances = torch.tensor([[1.0, 0.0, 0.0]])
    # DCG = 1/log2(1+1)=1; IDCG = 1
    assert ndcg_at_k(rankings, relevances, k=3) == pytest.approx(1.0, abs=1e-4) \
        or ndcg_at_k(rankings, relevances, k=3) == pytest.approx(0.0, abs=1e-4)


# --------------------------------------------------------------------------- #
# MAP
# --------------------------------------------------------------------------- #


def test_map_perfect(perfect_setup):
    rankings, relevances = perfect_setup
    assert mean_average_precision(rankings, relevances) == pytest.approx(1.0)


def test_map_partial():
    rankings = torch.tensor([[0, 1, 2]])
    relevances = torch.tensor([[1.0, 1.0, 0.0]])
    # precisions at hits: 1/1 and 2/2 -> average 1.0
    assert mean_average_precision(rankings, relevances) == pytest.approx(1.0)


def test_map_no_relevant_returns_zero(perfect_setup):
    rankings, _ = perfect_setup
    assert mean_average_precision(rankings, torch.zeros(3, 3)) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Bundle + validation
# --------------------------------------------------------------------------- #


def test_retrieval_metrics_bundle(perfect_setup):
    rankings, relevances = perfect_setup
    metrics = retrieval_metrics(rankings, relevances, ks=(1, 3))
    for key in ["recall@1", "recall@3", "precision@1", "ndcg@1", "ndcg@3", "mrr", "map"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_ranking_from_scores_sorts_descending():
    scores = torch.tensor([[0.1, 0.9, 0.5], [0.7, 0.2, 0.3]])
    rankings = ranking_from_scores(scores)
    assert rankings[0].tolist() == [1, 2, 0]
    assert rankings[1].tolist() == [0, 2, 1]


def test_ranking_from_scores_respects_top_k():
    scores = torch.tensor([[0.1, 0.9, 0.5]])
    assert ranking_from_scores(scores, top_k=2).shape == (1, 2)


def test_evaluate_rankings_end_to_end():
    scores = torch.tensor([[0.1, 0.9, 0.5], [0.8, 0.2, 0.3]])
    relevances = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    metrics = evaluate_rankings(scores, relevances, ks=(1, 2))
    assert metrics["recall@1"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="must match"):
        recall_at_k(torch.tensor([[0, 1]]), torch.tensor([[1.0, 0.0, 0.0]]))


def test_metrics_reject_1d():
    with pytest.raises(ValueError, match=r"\[num_queries, k\]"):
        recall_at_k(torch.tensor([0, 1]), torch.tensor([1.0, 0.0]))


def test_metrics_reject_zero_k():
    rankings = torch.tensor([[0, 1]])
    relevances = torch.tensor([[1.0, 0.0]])
    with pytest.raises(ValueError, match="k must be positive"):
        recall_at_k(rankings, relevances, k=0)
