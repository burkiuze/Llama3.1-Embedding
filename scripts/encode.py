#!/usr/bin/env python3
"""Encode texts with Llama3.1-Embedding and save or print the vectors.

    python scripts/encode.py \
        --backbone meta-llama/Llama-3.1-8B \
        --checkpoint runs/projection-768 \
        --input texts.txt --output embeddings.npy

    # single strings straight from the CLI
    python scripts/encode.py --checkpoint runs/projection-768 \
        --text "a search and rescue drone" "an emergency aircraft"

Model access: Meta's Llama 3.1 weights are gated. Accept the licence at
https://www.llama.com/llama3_1/ and authenticate locally (`huggingface-cli login`
or `HF_TOKEN`). Nothing is bundled with this repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_embedding.model import LlamaEmbeddingModel  # noqa: E402
from llama_embedding.tokenizer import build_tokenizer  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Encode texts into dense vectors")
    p.add_argument("--backbone", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--checkpoint", help="training output dir containing embedding_head.pt")
    p.add_argument("--text", nargs="+", help="one or more texts to encode")
    p.add_argument("--input", help="file with one text per line")
    p.add_argument("--output", help="path to save embeddings (.npy) or (.pt)")
    p.add_argument("--role", choices=["query", "document", None], default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--embedding-dim", type=int, default=None,
                   choices=[256, 384, 512, 768, 1024])
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16", "auto"])
    p.add_argument("--prefix-dim", type=int, default=None,
                   help="truncate to this many dims (Matryoshka read path)")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--smoke-test", action="store_true", help="stub backbone, no download")
    return p.parse_args(argv)


def read_texts(args):
    if args.text:
        return list(args.text)
    if args.input:
        with open(args.input, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    raise SystemExit("provide --text or --input")


def main(argv=None):
    args = parse_args(argv)
    texts = read_texts(args)

    if args.smoke_test:
        from llama_embedding.config import EmbeddingConfig
        from llama_embedding.model import StubBackbone
        from llama_embedding.tokenizer import DummyTokenizer

        config = EmbeddingConfig(
            backbone_name_or_path="stub://tiny", hidden_size=64,
            embedding_dim=args.embedding_dim or 768, max_length=32, dtype="float32",
        )
        model = LlamaEmbeddingModel(
            backbone=StubBackbone(hidden_size=64, seed=0),
            config=config,
            tokenizer=DummyTokenizer(),
        )
        print("[smoke-test] using the stub backbone; vectors are meaningless numerically.")
    else:
        overrides = {}
        if args.embedding_dim is not None:
            overrides["embedding_dim"] = args.embedding_dim
        model = LlamaEmbeddingModel.from_pretrained(
            args.backbone,
            output_dir=args.checkpoint,
            device=args.device,
            dtype=args.dtype,
            gradient_checkpointing=False,
            **overrides,
        )
        if model.tokenizer is None:
            model.tokenizer = build_tokenizer(model.config.backbone_name_or_path)

    vectors = model.encode(
        texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        normalize=False if args.no_normalize else None,
        prompt_role=args.role,
    )

    if args.prefix_dim:
        vectors = model.head.truncate(vectors, args.prefix_dim)

    print(f"encoded {vectors.shape[0]} texts -> shape {tuple(vectors.shape)}")
    print(f"norms: {[round(float(v), 4) for v in vectors.norm(dim=-1)][:10]}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        if args.output.endswith(".npy"):
            try:
                import numpy as np

                np.save(args.output, vectors.cpu().numpy())
            except ImportError:
                torch.save(vectors.cpu(), args.output)
                print(f"[numpy unavailable] saved as torch tensor to {args.output}")
        else:
            torch.save(vectors.cpu(), args.output)
        print(f"saved -> {args.output}")
    else:
        # Print a compact JSON preview of the first few vectors.
        preview = vectors[:3].tolist()
        print(json.dumps(preview, indent=2)[:800])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
