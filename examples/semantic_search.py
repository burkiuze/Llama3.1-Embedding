#!/usr/bin/env python3
"""Semantic search over a small corpus — the end-to-end retrieval demo.

    python examples/semantic_search.py --checkpoint runs/projection-768

    # CPU smoke run, no weights downloaded
    python examples/semantic_search.py --smoke-test

    # vector-store dependencies are optional; this example uses plain cosine
    # similarity. For FAISS/Qdrant/Chroma, keep the encoder and swap the index.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOCUMENTS = [
    "The search and rescue team located two hikers near the ridge at dusk.",
    "Rescue drones can thermal-imagine dense forest where people are lost.",
    "A thermal camera detects body heat through smoke and darkness.",
    "Paris is the capital and most populous city of France.",
    "The French Republic is a country in Western Europe.",
    "Mount Everest is the highest mountain above sea level at 8,849 metres.",
    "K2 is the second-highest mountain on Earth.",
    "list.append() adds an element to the end of a Python list.",
    "Python is a high-level programming language created by Guido van Rossum.",
]

QUERIES = [
    "finding missing people with a drone",
    "what is the tallest mountain",
    "adding something to a python list",
    "how to cook pasta",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Semantic search demo")
    parser.add_argument("--backbone", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--checkpoint", help="trained output dir")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--query", action="append", default=None,
                        help="repeatable; overrides the built-in queries")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)

    from llama_embedding.inference import SemanticSearchIndex

    if args.smoke_test:
        from llama_embedding.config import EmbeddingConfig
        from llama_embedding.model import LlamaEmbeddingModel, StubBackbone
        from llama_embedding.tokenizer import DummyTokenizer

        print("[smoke-test] stub backbone — the *plumbing* below is real, the "
              "ranking quality is not.\n")
        model = LlamaEmbeddingModel(
            backbone=StubBackbone(hidden_size=64, seed=0),
            config=EmbeddingConfig(backbone_name_or_path="stub://tiny", hidden_size=64,
                                   embedding_dim=768, max_length=32, dtype="float32"),
            tokenizer=DummyTokenizer(),
        )
    else:
        from llama_embedding.model import LlamaEmbeddingModel
        from llama_embedding.tokenizer import build_tokenizer

        model = LlamaEmbeddingModel.from_pretrained(args.backbone, output_dir=args.checkpoint)
        if model.tokenizer is None:
            model.tokenizer = build_tokenizer(model.config.backbone_name_or_path)

    print(f"Indexing {len(DOCUMENTS)} documents "
          f"(dim={model.config.embedding_dim}, pooling={model.config.pooling})...\n")
    index = SemanticSearchIndex.build(model, DOCUMENTS, batch_size=4)

    queries = args.query or QUERIES
    for query in queries:
        query_vector = model.encode([query], prompt_role="query")
        hits = index.search(query_vector, top_k=args.top_k)
        print(f'Query: "{query}"')
        for rank, hit in enumerate(hits, start=1):
            print(f"  {rank}. [{hit.score:+.4f}] {hit.text}")
        print()

    print("Swap-in points for a real vector store:")
    print("  - keep `model.encode` as the embedding function")
    print("  - replace SemanticSearchIndex with faiss.IndexFlatIP / Qdrant / Chroma")
    print("  - because vectors are unit-norm, inner product == cosine similarity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
