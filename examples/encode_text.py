#!/usr/bin/env python3
"""Encode a few sentences and inspect the vectors.

    # with a trained checkpoint
    python examples/encode_text.py --checkpoint runs/projection-768

    # CPU smoke run, no weights downloaded
    python examples/encode_text.py --smoke-test

    # gated Llama access: accept the licence at https://www.llama.com/llama3_1/
    # then authenticate with `huggingface-cli login` or set HF_TOKEN.
    python examples/encode_text.py --backbone meta-llama/Llama-3.1-8B
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEXTS = [
    "A search and rescue drone",
    "An unmanned aircraft used for emergency response",
    "A recipe for chocolate chip cookies",
    "Instructions for assembling a bicycle",
]


def main(argv=None):
    import torch

    parser = argparse.ArgumentParser(description="Encode example sentences")
    parser.add_argument("--backbone", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--checkpoint", help="trained output dir")
    parser.add_argument("--embedding-dim", type=int, default=768,
                        choices=[256, 384, 512, 768, 1024])
    parser.add_argument("--prefix-dim", type=int, default=None,
                        help="truncate to N dims (Matryoshka read path)")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)

    from llama_embedding.similarity import cosine_similarity

    if args.smoke_test:
        from llama_embedding.config import EmbeddingConfig
        from llama_embedding.model import LlamaEmbeddingModel, StubBackbone
        from llama_embedding.tokenizer import DummyTokenizer

        print("[smoke-test] stub backbone — shape/normalisation are real, "
              "semantics are not.\n")
        model = LlamaEmbeddingModel(
            backbone=StubBackbone(hidden_size=64, seed=0),
            config=EmbeddingConfig(backbone_name_or_path="stub://tiny", hidden_size=64,
                                   embedding_dim=args.embedding_dim, max_length=32,
                                   dtype="float32"),
            tokenizer=DummyTokenizer(),
        )
    else:
        from llama_embedding.model import LlamaEmbeddingModel
        from llama_embedding.tokenizer import build_tokenizer

        model = LlamaEmbeddingModel.from_pretrained(args.backbone, output_dir=args.checkpoint)
        if model.tokenizer is None:
            model.tokenizer = build_tokenizer(model.config.backbone_name_or_path)

    vectors = model.encode(TEXTS, batch_size=4, max_length=64)
    if args.prefix_dim:
        vectors = model.head.truncate(vectors, args.prefix_dim)

    print(f"Encoded {len(TEXTS)} texts")
    print(f"  shape : {tuple(vectors.shape)}")
    print(f"  norms : {[round(float(n), 4) for n in vectors.norm(dim=-1)]}")

    print("\nPairwise cosine similarity:")
    sim = cosine_similarity(vectors, vectors)
    for i, text in enumerate(TEXTS):
        print(f"  {text[:38]:<40} " + " ".join(f"{v:+.3f}" for v in sim[i].tolist()))

    print("\nMost similar to 'A search and rescue drone':")
    scores = cosine_similarity(vectors[:1], vectors)[0]
    order = torch.argsort(scores, descending=True)
    for idx in order[:3].tolist():
        print(f"  {scores[idx]:+.4f}  {TEXTS[idx]}")

    print("\nNear-duplicate check (texts 0 and 1 are paraphrases):")
    print(f"  sim(text0, text1) = {float(sim[0, 1]):+.4f}")
    print(f"  sim(text0, text2) = {float(sim[0, 2]):+.4f}  (unrelated topic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
