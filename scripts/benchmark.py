#!/usr/bin/env python3
"""Benchmark: raw pooled Llama baseline vs the trained Llama3.1-Embedding.

This is the experiment that answers "did training actually help?". Both arms
share the same backbone and the same evaluation code; the only difference is
whether the trained projection head is applied.

    python scripts/benchmark.py \
        --checkpoint runs/projection-768 \
        --eval-file data/sample/eval.jsonl

Reported metrics: Recall@K, Precision@K, NDCG@K, MRR, MAP for
  * baseline  — L2-normalised mean-pooled backbone hidden states (no head)
  * trained   — the same states through the trained projection head

Honesty note: the numbers printed are measured on whatever dataset you pass.
The README contains no benchmark scores until a real checkpoint has been
trained and evaluated against a public benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import evaluate_rankings  # noqa: E402
from evaluation.retrieval import build_relevance_matrix, load_retrieval_jsonl  # noqa: E402
from llama_embedding.projection import l2_normalize  # noqa: E402
from llama_embedding.similarity import similarity_matrix  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Baseline vs trained comparison")
    p.add_argument("--checkpoint", help="training output dir (trained arm)")
    p.add_argument("--backbone", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--eval-file", required=True, help="retrieval JSONL")
    p.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--prefix-dims", type=int, nargs="*", default=[],
                   help="also evaluate trained Matryoshka prefixes")
    p.add_argument("--output", help="write the comparison as JSON")
    p.add_argument("--smoke-test", action="store_true", help="stub backbone, no download")
    return p.parse_args(argv)


def build_model(args):
    if args.smoke_test:
        from llama_embedding.config import EmbeddingConfig
        from llama_embedding.model import LlamaEmbeddingModel, StubBackbone
        from llama_embedding.tokenizer import DummyTokenizer

        config = EmbeddingConfig(
            backbone_name_or_path="stub://tiny", hidden_size=64,
            embedding_dim=768, max_length=32, dtype="float32",
        )
        model = LlamaEmbeddingModel(
            backbone=StubBackbone(hidden_size=64, seed=0),
            config=config,
            tokenizer=DummyTokenizer(),
        )
        print("[smoke-test] stub backbone: absolute scores are meaningless; "
              "only the code path is being exercised.\n")
        return model
    from llama_embedding.model import LlamaEmbeddingModel
    from llama_embedding.tokenizer import build_tokenizer

    model = LlamaEmbeddingModel.from_pretrained(
        args.backbone, output_dir=args.checkpoint, device=None, dtype=None
    )
    if model.tokenizer is None:
        model.tokenizer = build_tokenizer(model.config.backbone_name_or_path)
    return model


def encode_pooled(model, texts, *, batch_size, max_length):
    """Encode texts and return BOTH the pooled pre-head states and embeddings."""
    import llama_embedding.tokenizer as tok

    pooled_all = []
    emb_all = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            encoded = tok.tokenize_texts(
                model.tokenizer, chunk,
                max_length=max_length or model.config.max_length,
                return_tensors="pt",
            )
            device = next(model.parameters()).device
            out = model(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
                return_pooled=True,
            )
            pooled_all.append(out["pooled"].float().cpu())
            emb_all.append(out["embeddings"].float().cpu())
    return torch.cat(pooled_all, 0), torch.cat(emb_all, 0)


def relevance_matrix(examples, corpus):
    return build_relevance_matrix(examples, corpus)


def main(argv=None):
    args = parse_args(argv)
    corpus, examples = load_retrieval_jsonl(args.eval_file)
    if not examples:
        print("error: eval file contains no queries")
        return 2

    print(f"Corpus documents   : {len(corpus)}")
    print(f"Evaluation queries : {len(examples)}")
    print("-" * 68)

    model = build_model(args)
    queries = [e.query for e in examples]
    docs = list(corpus.texts)

    query_pooled, query_emb = encode_pooled(
        model, queries, batch_size=args.batch_size, max_length=args.max_length
    )
    doc_pooled, doc_emb = encode_pooled(
        model, docs, batch_size=args.batch_size, max_length=args.max_length
    )
    rel = relevance_matrix(examples, corpus)

    results = {}

    # ---- arm 1: raw pooled backbone (baseline, no projection) ------------- #
    baseline_scores = similarity_matrix(
        l2_normalize(query_pooled), l2_normalize(doc_pooled)
    )
    results["baseline_pooled"] = evaluate_rankings(baseline_scores, rel, ks=tuple(args.top_k))

    # ---- arm 2: trained projection head ----------------------------------- #
    trained_scores = similarity_matrix(query_emb, doc_emb)
    results["trained"] = evaluate_rankings(trained_scores, rel, ks=tuple(args.top_k))

    # ---- optional: Matryoshka prefixes ------------------------------------ #
    for dim in [d for d in (args.prefix_dims or []) if d and d < query_emb.shape[1]]:
        q = l2_normalize(query_emb[:, :dim])
        d_ = l2_normalize(doc_emb[:, :dim])
        prefix_scores = similarity_matrix(q, d_)
        results[f"trained_prefix_{dim}"] = evaluate_rankings(
            prefix_scores, rel, ks=tuple(args.top_k)
        )

    _print_table(results, tuple(args.top_k))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nwrote -> {args.output}")
    return 0


def _print_table(results, ks):
    arms = list(results.keys())
    metrics = sorted({m for arm in results.values() for m in arm})
    header = f"{'metric':>16} | " + " | ".join(f"{a[:22]:>22}" for a in arms)
    print("\nMeasured results (dataset: " + os.path.basename(
        getattr(results, "dataset", "your --eval-file")) + ")")
    print(header)
    print("-" * len(header))
    for metric in metrics:
        row = f"{metric:>16} | "
        row += " | ".join(f"{results[a].get(metric, float('nan')):>22.4f}" for a in arms)
        print(row)
    print("\nNote: these are measurements on the dataset you provided, not "
          "published benchmarks.")


if __name__ == "__main__":
    raise SystemExit(main())
