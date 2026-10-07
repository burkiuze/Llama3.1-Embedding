"""Similarity function tests: cosine, dot, euclidean and the similarity matrix."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from llama_embedding.projection import l2_normalize
from llama_embedding.similarity import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    similarity_matrix,
)


# --------------------------------------------------------------------------- #
# cosine_similarity
# --------------------------------------------------------------------------- #


def test_cosine_similarity_of_identical_vectors_is_one():
    x = torch.tensor([[3.0, 4.0]])
    assert torch.allclose(cosine_similarity(x, x), torch.tensor([1.0]), atol=1e-6)


def test_cosine_similarity_of_opposites_is_minus_one():
    x = torch.tensor([[1.0, 0.0]])
    y = torch.tensor([[-1.0, 0.0]])
    assert torch.allclose(cosine_similarity(x, y), torch.tensor([-1.0]), atol=1e-6)


def test_cosine_similarity_is_orthogonal_zero():
    x = torch.tensor([[1.0, 0.0]])
    y = torch.tensor([[0.0, 1.0]])
    assert torch.allclose(cosine_similarity(x, y), torch.tensor([0.0]), atol=1e-6)


def test_cosine_similarity_ignores_magnitude():
    """Scaling a vector must not change cosine similarity."""
    torch.manual_seed(0)
    a = torch.randn(6, 12)
    b = torch.randn(6, 12)
    direct = cosine_similarity(a, b)
    scaled = cosine_similarity(a * 7.5, b * 0.001)
    assert torch.allclose(direct, scaled, atol=1e-5)


def test_cosine_similarity_known_value():
    a = torch.tensor([[1.0, 1.0, 0.0]])
    b = torch.tensor([[1.0, 0.0, 0.0]])
    # cos angle between (1,1,0) and (1,0,0) = 1/sqrt(2)
    assert torch.allclose(cosine_similarity(a, b), torch.tensor([1.0 / math.sqrt(2)]), atol=1e-6)


def test_cosine_similarity_zero_vector_returns_zero():
    a = torch.zeros(1, 4)
    b = torch.ones(1, 4)
    out = cosine_similarity(a, b)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros(1))


def test_cosine_similarity_rejects_feature_mismatch():
    with pytest.raises(ValueError, match="feature dims differ"):
        cosine_similarity(torch.randn(2, 4), torch.randn(2, 5))


def test_cosine_similarity_symmetry():
    torch.manual_seed(1)
    a, b = torch.randn(5, 8), torch.randn(5, 8)
    assert torch.allclose(cosine_similarity(a, b), cosine_similarity(b, a), atol=1e-6)


# --------------------------------------------------------------------------- #
# dot / euclidean
# --------------------------------------------------------------------------- #


def test_dot_product_values():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    out = dot_product(a, b)
    assert torch.allclose(out, torch.tensor([17.0, 53.0]))


def test_dot_product_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="matching shapes"):
        dot_product(torch.randn(3, 4), torch.randn(2, 4))


def test_euclidean_distance_values():
    a = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    b = torch.tensor([[3.0, 4.0], [1.0, 1.0]])
    out = euclidean_distance(a, b)
    assert torch.allclose(out, torch.tensor([5.0, 0.0]), atol=1e-6)


# --------------------------------------------------------------------------- #
# similarity_matrix
# --------------------------------------------------------------------------- #


def test_similarity_matrix_shape():
    q = torch.randn(4, 16)
    d = torch.randn(7, 16)
    assert similarity_matrix(q, d).shape == (4, 7)


def test_similarity_matrix_diagonal_is_maximal_for_self():
    """A matched (query, its own document) set should retrieve itself first."""
    torch.manual_seed(7)
    docs = l2_normalize(torch.randn(6, 32))
    scores = similarity_matrix(docs, docs)
    assert torch.allclose(scores.diagonal(), torch.ones(6), atol=1e-5)
    assert torch.equal(scores.argmax(dim=1), torch.arange(6))


def test_similarity_matrix_cosine_equals_dot_on_normalised_vectors():
    torch.manual_seed(11)
    q = l2_normalize(torch.randn(5, 12))
    d = l2_normalize(torch.randn(9, 12))
    assert torch.allclose(
        similarity_matrix(q, d, metric="cosine"),
        similarity_matrix(q, d, metric="dot"),
        atol=1e-5,
    )


def test_similarity_matrix_bounds_for_cosine():
    scores = similarity_matrix(torch.randn(10, 20), torch.randn(10, 20), metric="cosine")
    assert scores.min() >= -1.0001
    assert scores.max() <= 1.0001


def test_similarity_matrix_euclidean_is_negative_distance():
    q = torch.tensor([[0.0, 0.0]])
    d = torch.tensor([[3.0, 4.0]])
    assert torch.allclose(similarity_matrix(q, d, metric="euclidean"), torch.tensor([[-5.0]]), atol=1e-5)


def test_similarity_matrix_rejects_bad_metric():
    with pytest.raises(ValueError, match="unsupported metric"):
        similarity_matrix(torch.randn(2, 4), torch.randn(3, 4), metric="manhattan")


def test_similarity_matrix_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dims differ"):
        similarity_matrix(torch.randn(2, 4), torch.randn(3, 8))


def test_similarity_matrix_rejects_1d_input():
    with pytest.raises(ValueError, match="must be 2-D"):
        similarity_matrix(torch.randn(8), torch.randn(3, 8))


def test_similarity_matrix_preserves_ranking_under_rescaling():
    """Cosine ranking is invariant to positive scaling of any document."""
    torch.manual_seed(21)
    q = torch.randn(3, 16)
    d = torch.randn(10, 16)
    scale = torch.rand(10, 1) + 0.5
    base = similarity_matrix(q, d)
    scaled = similarity_matrix(q, d * scale)
    assert torch.equal(base.argsort(dim=1), scaled.argsort(dim=1))


def test_similarity_matrix_gradients_flow():
    q = torch.randn(3, 8, requires_grad=True)
    d = torch.randn(4, 8, requires_grad=True)
    similarity_matrix(q, d).sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert d.grad is not None and torch.isfinite(d.grad).all()
