"""Retrieval evaluation: embed a corpus, rank it, score it.

    queries  -> embedding -> similarity vs corpus -> ranking -> metrics
    corpus   -> embedding

The :class:`RetrievalEvaluator` works on a plain relevance matrix so it can be
driven by any encoder (the trained model, the raw pooled baseline, or
precomputed vectors from an external system), which is what makes the
before/after comparison in ``scripts/benchmark.py`` fair: only the embeddings
change, the evaluation code is identical.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from llama_embedding.similarity import similarity_matrix

from .metrics import ranking_from_scores, retrieval_metrics

__all__ = [
    "RetrievalExample",
    "RetrievalCorpus",
    "RetrievalEvaluator",
    "evaluate_retrieval",
    "load_retrieval_jsonl",
    "build_relevance_matrix",
]


@dataclass
class RetrievalExample:
    """One evaluation query with its relevant document ids."""

    query: str
    relevant_ids: List[str]
    id: Optional[str] = None


@dataclass
class RetrievalCorpus:
    """A document collection with stable ids."""

    ids: List[str]
    texts: List[str]

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.texts):
            raise ValueError("corpus ids and texts must have the same length")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("corpus ids must be unique")

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def id_to_index(self) -> Dict[str, int]:
        return {doc_id: i for i, doc_id in enumerate(self.ids)}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"RetrievalCorpus(size={len(self.ids)})"


def build_relevance_matrix(
    examples: Sequence[RetrievalExample], corpus: RetrievalCorpus
) -> torch.Tensor:
    """Build the ``[num_queries, num_docs]`` binary relevance matrix.

    Raises ``KeyError`` if a query references a document id absent from the
    corpus, which is a data bug worth surfacing loudly rather than silently
    scoring against an all-zero row.
    """
    relevances = torch.zeros(len(examples), len(corpus))
    id_to_index = corpus.id_to_index
    for i, example in enumerate(examples):
        for doc_id in example.relevant_ids:
            idx = id_to_index.get(doc_id)
            if idx is None:
                raise KeyError(f"query {example.query!r} references unknown document id {doc_id!r}")
            relevances[i, idx] = 1.0
    return relevances


@dataclass
class RetrievalResult:
    """Metrics plus the per-query rankings, ready to inspect or serialise."""

    metrics: Dict[str, float]
    rankings: torch.Tensor
    scores: torch.Tensor
    corpus_ids: List[str] = field(default_factory=list)
    query_ids: List[Optional[str]] = field(default_factory=list)

    def top_k(self, k: int, query_index: int = 0) -> List[Tuple[str, float]]:
        """Return ``(document_id, score)`` for the top ``k`` hits of one query."""
        idx = self.rankings[query_index][:k].tolist()
        sc = self.scores[query_index][idx].tolist()
        return [
            (self.corpus_ids[i] if i < len(self.corpus_ids) else str(i), float(s))
            for i, s in zip(idx, sc)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {"metrics": self.metrics, "num_queries": int(self.rankings.shape[0])}


class RetrievalEvaluator:
    """Score a query set against a corpus given an ``encode`` callable.

    Parameters
    ----------
    encode:
        ``callable(texts: list[str]) -> Tensor [len(texts), D]``.
    ks:
        Cut-offs to report metrics for.
    """

    def __init__(self, encode: Callable[[Sequence[str]], torch.Tensor], ks: Sequence[int] = (1, 5, 10)) -> None:
        self.encode = encode
        self.ks = tuple(ks)

    # -- encoding helpers ------------------------------------------------------ #
    def encode_corpus(self, corpus: RetrievalCorpus, *, batch_size: int = 8) -> torch.Tensor:
        return self.encode(list(corpus.texts))

    def encode_queries(self, examples: Sequence[RetrievalExample], *, batch_size: int = 8) -> torch.Tensor:
        return self.encode([e.query for e in examples])

    # -- core ------------------------------------------------------------------ #
    def evaluate(
        self,
        examples: Sequence[RetrievalExample],
        corpus: RetrievalCorpus,
        *,
        corpus_embeddings: Optional[torch.Tensor] = None,
        query_embeddings: Optional[torch.Tensor] = None,
        metric: str = "cosine",
    ) -> RetrievalResult:
        """Embed, rank and score. Any precomputed embeddings are reused."""
        if not examples:
            raise ValueError("no queries to evaluate")
        if len(corpus) == 0:
            raise ValueError("corpus is empty")

        corpus_emb = (
            corpus_embeddings
            if corpus_embeddings is not None
            else self.encode_corpus(corpus)
        )
        query_emb = (
            query_embeddings
            if query_embeddings is not None
            else self.encode_queries(examples)
        )
        corpus_emb = corpus_emb.float()
        query_emb = query_emb.float()

        if corpus_emb.shape[0] != len(corpus):
            raise ValueError(
                f"corpus embeddings have {corpus_emb.shape[0]} rows but the corpus has {len(corpus)} docs"
            )
        if query_emb.shape[0] != len(examples):
            raise ValueError("query embedding count does not match the number of queries")
        if corpus_emb.shape[1] != query_emb.shape[1]:
            raise ValueError(
                f"dimension mismatch: corpus {corpus_emb.shape[1]} vs queries {query_emb.shape[1]}"
            )

        scores = similarity_matrix(query_emb, corpus_emb, metric=metric)

        relevances = build_relevance_matrix(examples, corpus)

        depth = max(self.ks) if self.ks else relevances.shape[1]
        rankings = ranking_from_scores(scores, top_k=min(depth, relevances.shape[1]))
        metrics = retrieval_metrics(rankings, relevances, ks=self.ks)

        return RetrievalResult(
            metrics=metrics,
            rankings=rankings,
            scores=scores,
            corpus_ids=list(corpus.ids),
            query_ids=[e.id for e in examples],
        )


def evaluate_retrieval(
    examples: Sequence[RetrievalExample],
    corpus: RetrievalCorpus,
    encode: Callable[[Sequence[str]], torch.Tensor],
    *,
    ks: Sequence[int] = (1, 5, 10),
    metric: str = "cosine",
) -> RetrievalResult:
    """Convenience wrapper around :class:`RetrievalEvaluator`."""
    return RetrievalEvaluator(encode, ks=ks).evaluate(examples, corpus, metric=metric)


def load_retrieval_jsonl(path: str) -> Tuple[RetrievalCorpus, List[RetrievalExample]]:
    """Load a BEIR-style JSONL (``corpus``/``queries``/``qrels``) or a simple format.

    Supported per line:
        ``{"id": "d1", "text": "..."}``                        -> corpus
        ``{"_id": "d1", "title": "...", "text": "..."}``       -> corpus (title + text)
        ``{"query_id": "q1", "query": "..."}``                 -> query
        ``{"query_id": "q1", "relevant_ids": ["d1"]}``         -> query relevance

    Returns ``(corpus, examples)`` where each example's ``query`` defaults to its
    id when no text is present.
    """
    corpus_records: Dict[str, str] = {}
    query_texts: Dict[str, str] = {}
    relevance: Dict[str, List[str]] = {}

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "query_id" in record:
                qid = str(record["query_id"])
                if "query" in record:
                    query_texts[qid] = record["query"]
                rel = record.get("relevant_ids")
                if rel is None:
                    pos = record.get("positive_ids") or record.get("gold_ids")
                    rel = list(pos) if pos else []
                relevance[qid] = [str(r) for r in rel]
                continue
            doc_id = record.get("id") or record.get("_id") or record.get("doc_id")
            if doc_id is None:
                continue
            text = record.get("text") or ""
            title = record.get("title")
            corpus_records[str(doc_id)] = f"{title}\n{text}".strip() if title else text

    corpus = RetrievalCorpus(ids=list(corpus_records.keys()), texts=list(corpus_records.values()))
    examples = [
        RetrievalExample(query=query_texts.get(qid, qid), relevant_ids=rel, id=qid)
        for qid, rel in relevance.items()
        if qid in query_texts
    ]
    if not examples:
        examples = [
            RetrievalExample(query=qid, relevant_ids=rel, id=qid)
            for qid, rel in relevance.items()
        ]
    return corpus, examples
