"""Tests for the inference helpers: SemanticSearchIndex and ranking.

The index is exercised with synthetic unit vectors so the expected ranking is
known exactly, independently of any model's quality.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from llama_embedding.inference import SemanticSearchIndex, SearchResult, rank_documents
from llama_embedding.projection import l2_normalize


DOCS = [
    "a rescue drone finds missing hikers",
    "paris is the capital of france",
    "list.append adds an item",
]


def _vectors():
    """Three clearly separated, hand-built unit vectors."""
    e0 = torch.tensor([1.0, 0.0, 0.0])
    e1 = torch.tensor([0.0, 1.0, 0.0])
    e2 = torch.tensor([0.0, 0.0, 1.0])
    return torch.stack([e0, e1, e2])


def test_index_requires_documents_or_embeddings():
    with pytest.raises(ValueError, match="provide documents"):
        SemanticSearchIndex()


def test_index_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        SemanticSearchIndex(documents=DOCS, embeddings=torch.randn(2, 3))


def test_index_rejects_1d_embeddings():
    with pytest.raises(ValueError, match=r"\[N, D\]"):
        SemanticSearchIndex(documents=DOCS, embeddings=torch.randn(3))


def test_search_returns_flat_list_for_single_query():
    index = SemanticSearchIndex(documents=DOCS, embeddings=_vectors())
    hits = index.search(torch.tensor([[1.0, 0.0, 0.0]]), top_k=1)
    assert isinstance(hits, list)
    assert isinstance(hits[0], SearchResult)
    assert hits[0].index == 0
    assert hits[0].text == DOCS[0]


def test_search_accepts_1d_query_vector():
    index = SemanticSearchIndex(documents=DOCS, embeddings=_vectors())
    hits = index.search(torch.tensor([0.0, 0.0, 1.0]), top_k=1)
    assert hits[0].index == 2


def test_search_returns_nested_for_batch():
    index = SemanticSearchIndex(documents=DOCS, embeddings=_vectors())
    queries = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    hits = index.search(queries, top_k=1)
    assert len(hits) == 2
    assert hits[0][0].index == 0
    assert hits[1][0].index == 1


def test_search_ranks_correctly():
    index = SemanticSearchIndex(documents=DOCS, embeddings=_vectors())
    hits = index.search(torch.tensor([[1.0, 0.1, 0.0]]), top_k=3)
    assert [h.index for h in hits] == [0, 1, 2]


def test_search_caps_top_k_at_corpus_size():
    index = SemanticSearchIndex(documents=DOCS, embeddings=_vectors())
    assert len(index.search(torch.tensor([[1.0, 0.0, 0.0]]), top_k=99)) == 3


def test_search_rejects_non_positive_top_k():
    index = SemanticSearchIndex(documents=DOCS, embeddings=_vectors())
    with pytest.raises(ValueError, match="top_k must be positive"):
        index.search(torch.tensor([[1.0, 0.0, 0.0]]), top_k=0)


def test_search_requires_precomputed_vectors():
    index = SemanticSearchIndex(documents=DOCS, embeddings=_vectors())
    with pytest.raises(ValueError, match="precomputed query vectors"):
        index.search("some raw string")


def test_search_on_empty_index_raises():
    index = SemanticSearchIndex(documents=[], embeddings=torch.zeros(0, 3))
    with pytest.raises(RuntimeError, match="no embeddings"):
        index.search(torch.tensor([[1.0, 0.0, 0.0]]))


def test_index_normalises_stored_embeddings():
    """Vectors are L2-normalised on ingest so scoring is scale-invariant."""
    scaled = _vectors() * 17.0
    index = SemanticSearchIndex(documents=DOCS, embeddings=scaled)
    assert torch.allclose(index.embeddings.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_search_scores_are_cosine_in_range():
    index = SemanticSearchIndex(documents=DOCS, embeddings=_vectors() * 5)
    hits = index.search(torch.tensor([[1.0, 1.0, 0.0]]), top_k=3)
    for hit in hits:
        assert -1.0001 <= hit.score <= 1.0001


# --------------------------------------------------------------------------- #
# rank_documents
# --------------------------------------------------------------------------- #


def test_rank_documents_orders_by_similarity():
    docs = _vectors()
    ranked = rank_documents(torch.tensor([0.0, 1.0, 0.0]), docs, top_k=3)
    assert ranked[0][0] == 1
    assert ranked == sorted(ranked, key=lambda kv: kv[1], reverse=True)


def test_rank_documents_accepts_2d_query():
    ranked = rank_documents(torch.tensor([[1.0, 0.0, 0.0]]), _vectors(), top_k=1)
    assert ranked[0][0] == 0


def test_rank_documents_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension mismatch"):
        rank_documents(torch.randn(5), _vectors())


def test_rank_documents_rejects_bad_query_rank():
    with pytest.raises(ValueError, match=r"\[D\] or \[1, D\]"):
        rank_documents(torch.randn(2, 2, 5), _vectors())


def test_rank_documents_rejects_bad_doc_rank():
    with pytest.raises(ValueError, match=r"\[N, D\]"):
        rank_documents(torch.randn(3), torch.randn(3, 3, 1))


def test_rank_documents_is_scale_invariant():
    docs = _vectors()
    a = rank_documents(torch.tensor([1.0, 1.0, 0.0]), docs)
    b = rank_documents(torch.tensor([1.0, 1.0, 0.0]) * 9.0, docs * 0.3)
    assert [i for i, _ in a] == [i for i, _ in b]
    assert [s for _, s in a] == pytest.approx([s for _, s in b], abs=1e-5)
