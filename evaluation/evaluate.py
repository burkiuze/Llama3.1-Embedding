#!/usr/bin/env python3
"""Evaluate a Llama3.1-Embedding checkpoint (or the untrained baseline).

    python -m evaluation.evaluate \
        --checkpoint runs/projection-768 \
        --eval-file data/sample/eval.jsonl \
        --top-k 1 5 10

This reports **measured** numbers for the data you point it at. The repository
ships no benchmark scores: every table in the README says NOT YET BENCHMARKED
until a real checkpoint has been trained and run against a public benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.retrieval import (  # noqa: E402
    RetrievalEvaluator,
    build_relevance_matrix,
    load_retrieval_jsonl,
)
from llama_embedding.model import LlamaEmbeddingModel, StubBackbone  # noqa: E402
from llama_embedding.tokenizer import DummyTokenizer, build_tokenizer  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Llama3.1-Embedding retrieval quality")
    parser.add_argument("--checkpoint", help="directory produced by training (embedding_head.pt)")
    parser.add_argument("--backbone", default=None, help="HF model id/path for the base Llama 3.1 model")
    parser.add_argument("--eval-file", required=True, help="retrieval JSONL (corpus + queries + qrels)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--prefix-dims", type=int, nargs="*", default=[], help="Matryoshka prefix dims to evaluate")
    parser.add_argument("--output", help="write metrics as JSON to this path")
    parser.add_argument("--smoke-test", action="store_true", help="use the stub backbone (no download)")
    return parser.parse_args(argv)


def _build_model(args: argparse.Namespace) -> LlamaEmbeddingModel:
    if args.smoke_test:
        from llama_embedding.config import EmbeddingConfig

        config = EmbeddingConfig(
            backbone_name_or_path="stub://tiny",
            hidden_size=64,
            embedding_dim=768,
            max_length=32,
            dtype="float32",
        )
        return LlamaEmbeddingModel(
            backbone=StubBackbone(hidden_size=64, seed=0),
            config=config,
            tokenizer=DummyTokenizer(),
        )
    model = LlamaEmbeddingModel.from_pretrained(
        args.backbone,
        output_dir=args.checkpoint,
        device=None,
        dtype=None,
    )
    if model.tokenizer is None:
        # from_pretrained only auto-loads a tokenizer when a backbone id was
        # passed explicitly; resolve it from the stored config otherwise.
        model.tokenizer = build_tokenizer(model.config.backbone_name_or_path)
    return model


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    corpus, examples = load_retrieval_jsonl(args.eval_file)
    print(f"Corpus documents : {len(corpus)}")
    print(f"Evaluation queries: {len(examples)}")
    if not examples:
        print("error: no queries with relevance judgements found in the eval file")
        return 2

    model = _build_model(args)
    evaluator = RetrievalEvaluator(
        lambda texts: model.encode(
            texts, batch_size=args.batch_size, max_length=args.max_length
        ),
        ks=tuple(args.top_k),
    )

    prefix_dims: List[int] = [d for d in (args.prefix_dims or []) if d]
    all_metrics = {}

    base_result = evaluator.evaluate(examples, corpus, metric="cosine")
    all_metrics[str(model.config.embedding_dim)] = base_result.metrics
    _print_metrics(base_result.metrics, title=f"full dim={model.config.embedding_dim}")

    if prefix_dims:
        from evaluation.metrics import evaluate_rankings
        from llama_embedding.projection import l2_normalize
        from llama_embedding.similarity import similarity_matrix

        query_emb = evaluator.encode_queries(examples)
        corpus_emb = evaluator.encode_corpus(corpus)
        relevances = build_relevance_matrix(examples, corpus)
        for dim in prefix_dims:
            q = l2_normalize(query_emb.float()[:, :dim])
            c = l2_normalize(corpus_emb.float()[:, :dim])
            scores = similarity_matrix(q, c)
            metrics = evaluate_rankings(scores, relevances, ks=tuple(args.top_k))
            all_metrics[str(dim)] = metrics
            _print_metrics(metrics, title=f"matryoshka prefix dim={dim}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(all_metrics, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nWrote metrics to {args.output}")
    return 0


def _print_metrics(metrics, *, title: str) -> None:
    print(f"\n--- {title} ---")
    for key in sorted(metrics):
        print(f"  {key:>16}: {metrics[key]:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
