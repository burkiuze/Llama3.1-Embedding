"""Production-style inference helpers: semantic search over encoded corpora.

Kept dependency-light on purpose: the vector-store part is a small in-memory
index so the example runs anywhere, while ``faiss``/``qdrant``/``chroma``
remain entirely optional integrations the user can layer on top of
:class:`SemanticSearchIndex`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from .model import LlamaEmbeddingModel
from .projection import l2_normalize
from .similarity import cosine_similarity

__all__ = ["SemanticSearchIndex", "cosine_similarity", "rank_documents"]


@dataclass
class SearchResult:
    """One ranked document."""

    index: int
    score: float
    text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        preview = (self.text or "")[:60]
        return f"SearchResult(index={self.index}, score={self.score:.4f}, text={preview!r})"


class SemanticSearchIndex:
    """A minimal in-memory cosine-similarity index.

    Because the encoder emits unit-norm vectors, cosine similarity equals a
    normalised dot product, so retrieval is a single matrix product.

    Example
    -------
    >>> index = SemanticSearchIndex(documents)          # doctest: +SKIP
    >>> index.search("find the drone", top_k=2)         # doctest: +SKIP
    """

    def __init__(
        self,
        documents: Optional[Sequence[str]] = None,
        embeddings: Optional[torch.Tensor] = None,
        *,
        texts: Optional[Sequence[str]] = None,
    ) -> None:
        if embeddings is None and documents is None and texts is None:
            raise ValueError("provide documents, texts or precomputed embeddings")
        corpus = documents if documents is not None else texts
        if embeddings is not None and embeddings.dim() != 2:
            raise ValueError(f"embeddings must be [N, D]; got {tuple(embeddings.shape)}")
        if embeddings is not None and corpus is not None:
            if len(corpus) != embeddings.shape[0]:
                raise ValueError(
                    f"documents ({len(corpus)}) and embeddings ({embeddings.shape[0]}) "
                    "must have the same length"
                )
        self.documents: List[str] = list(corpus) if corpus is not None else []
        self.embeddings: Optional[torch.Tensor] = (
            l2_normalize(embeddings.float()) if embeddings is not None else None
        )

    # -- building ------------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        model: LlamaEmbeddingModel,
        documents: Sequence[str],
        *,
        batch_size: int = 8,
        max_length: Optional[int] = None,
        prompt_role: Optional[str] = "document",
    ) -> "SemanticSearchIndex":
        """Encode ``documents`` with ``model`` and return a populated index."""
        vectors = model.encode(
            documents, batch_size=batch_size, max_length=max_length, prompt_role=prompt_role
        )
        return cls(documents=documents, embeddings=vectors)

    # -- querying ------------------------------------------------------------- #
    def search(
        self,
        queries: Union[str, torch.Tensor, Sequence[str]],
        *,
        top_k: int = 5,
        batch_size: int = 8,
        max_length: Optional[int] = None,
        query_vectors: Optional[torch.Tensor] = None,
    ) -> Union[List[SearchResult], List[List[SearchResult]]]:
        """Return the ``top_k`` most similar documents per query."""
        if self.embeddings is None or self.embeddings.numel() == 0:
            raise RuntimeError(
                "index holds no embeddings; build it with SemanticSearchIndex.build()"
            )
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        k = min(top_k, self.embeddings.shape[0])

        # Decide the return shape *before* `queries` is consumed: a single
        # string, a [D] vector or a [1, D] batch yields a flat list of results;
        # anything else yields one list per query.
        if isinstance(queries, torch.Tensor):
            query_vectors = queries
        single = isinstance(queries, str) or (
            query_vectors is not None
            and query_vectors.dim() in (1, 2)
            and query_vectors.shape[0] == 1
        )
        if query_vectors is None:
            raise ValueError(
                "search() needs precomputed query vectors: pass query_vectors=<Tensor> "
                "or a torch.Tensor directly. Use SemanticSearchIndex.build() + "
                "model.encode() to produce them."
            )
        if query_vectors.dim() == 1:
            query_vectors = query_vectors.unsqueeze(0)

        q = l2_normalize(query_vectors.float())
        scores = q @ self.embeddings.T  # [Q, N]
        top_scores, top_indices = torch.topk(scores, k=k, dim=-1)
        results: List[List[SearchResult]] = []
        for row_scores, row_indices in zip(top_scores, top_indices):
            row: List[SearchResult] = []
            for score, idx in zip(row_scores.tolist(), row_indices.tolist()):
                row.append(
                    SearchResult(
                        index=idx,
                        score=float(score),
                        text=self.documents[idx] if idx < len(self.documents) else None,
                    )
                )
            results.append(row)
        return results[0] if single else results


def rank_documents(
    query_vector: torch.Tensor,
    document_vectors: torch.Tensor,
    *,
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """Rank precomputed ``document_vectors`` against one ``query_vector``.

    Returns ``[(index, score), ...]`` sorted by descending cosine similarity.
    """
    if query_vector.dim() == 2 and query_vector.shape[0] == 1:
        query_vector = query_vector.squeeze(0)
    if query_vector.dim() != 1:
        raise ValueError(f"query_vector must be [D] or [1, D]; got {tuple(query_vector.shape)}")
    if document_vectors.dim() != 2:
        raise ValueError(f"document_vectors must be [N, D]; got {tuple(document_vectors.shape)}")
    if query_vector.shape[0] != document_vectors.shape[1]:
        raise ValueError(
            f"dimension mismatch: query {query_vector.shape[0]} vs docs {document_vectors.shape[1]}"
        )
    scores = cosine_similarity(query_vector.unsqueeze(0), document_vectors).squeeze(0)
    k = min(top_k, scores.shape[0])
    values, indices = torch.topk(scores, k=k)
    return [(int(i), float(s)) for s, i in zip(values.tolist(), indices.tolist())]
