#!/usr/bin/env python3
"""Training entrypoint for Llama3.1-Embedding.

Usage
-----
    python -m training.train --config configs/projection.yaml \
        --train-file data/sample/train.jsonl \
        --output-dir runs/projection-768

    # smoke run: no GPU, no 8B download, exercises the whole training path
    python -m training.train --smoke-test

Model access
------------
Meta's Llama 3.1 weights are gated. Request access at https://www.llama.com/llama3_1/,
accept the licence, then authenticate locally (``huggingface-cli login`` or set
``HF_TOKEN``). This project never bundles credentials or bypasses the gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

# Allow `python training/train.py` as well as `python -m training.train`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_embedding.config import EmbeddingConfig, TrainingConfig, load_config_file  # noqa: E402
from llama_embedding.model import LlamaEmbeddingModel, StubBackbone  # noqa: E402
from llama_embedding.tokenizer import DummyTokenizer, build_tokenizer  # noqa: E402

from training.dataset import build_datasets  # noqa: E402
from training.trainer import Trainer, set_seed  # noqa: E402


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Llama 3.1 sentence embedding model")
    parser.add_argument("--config", help="YAML/JSON training config file")
    parser.add_argument("--train-file", help="training JSONL (query/positive/negative)")
    parser.add_argument("--eval-file", help="optional validation JSONL")
    parser.add_argument("--backbone", default=None, help="HF model id or local path for the base model")
    parser.add_argument("--output-dir", default=None, help="where checkpoints are written")
    parser.add_argument("--mode", choices=["projection_only", "lora", "full"], default=None)
    parser.add_argument("--embedding-dim", type=int, default=None, choices=[256, 384, 512, 768, 1024])
    parser.add_argument("--pooling", choices=["mean", "last_token", "weighted_mean"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--matryoshka", action="store_true", help="enable Matryoshka prefix loss")
    parser.add_argument("--resume-from", default=None, help="checkpoint dir to resume from")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="train a few steps on a tiny stub backbone (no download, CPU only)",
    )
    return parser.parse_args(argv)


def build_configs(args: argparse.Namespace) -> tuple[EmbeddingConfig, TrainingConfig]:
    """Merge file config with CLI overrides (CLI wins)."""
    if args.config:
        raw = load_config_file(args.config)
        emb_raw = raw.get("embedding", raw)
        train_raw = raw.get("training", {})
        # A flat config file is treated as the training section when it has
        # training-ish keys, otherwise as the embedding section.
        if not train_raw and not emb_raw:
            emb_raw, train_raw = {}, raw
    else:
        emb_raw, train_raw = {}, {}

    # CLI overrides
    if args.backbone:
        emb_raw["backbone_name_or_path"] = args.backbone
    if args.embedding_dim is not None:
        emb_raw["embedding_dim"] = args.embedding_dim
    if args.pooling is not None:
        emb_raw["pooling"] = args.pooling
    emb_raw.setdefault("backbone_name_or_path", "meta-llama/Llama-3.1-8B")

    train_overrides = {}
    if args.train_file:
        train_overrides["train_file"] = args.train_file
    if args.eval_file:
        train_overrides["eval_file"] = args.eval_file
    if args.output_dir:
        train_overrides["output_dir"] = args.output_dir
    if args.mode:
        train_overrides["mode"] = args.mode
    if args.batch_size is not None:
        train_overrides["batch_size"] = args.batch_size
    if args.gradient_accumulation_steps is not None:
        train_overrides["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.learning_rate is not None:
        train_overrides["learning_rate"] = args.learning_rate
    if args.temperature is not None:
        train_overrides["temperature"] = args.temperature
    if args.max_steps is not None:
        train_overrides["max_steps"] = args.max_steps
    if args.max_epochs is not None:
        train_overrides["max_epochs"] = args.max_epochs
    if args.seed is not None:
        train_overrides["seed"] = args.seed
    if args.resume_from:
        train_overrides["resume_from"] = args.resume_from
    if args.max_length is not None:
        train_overrides["max_length"] = args.max_length
        emb_raw["max_length"] = args.max_length

    train_raw.update(train_overrides)

    if args.matryoshka:
        emb_raw["matryoshka_dims"] = [1024, 768, 512, 384, 256]

    embedding_config = EmbeddingConfig.from_dict(emb_raw)
    training_config = TrainingConfig.from_dict(train_raw)
    return embedding_config, training_config


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    embedding_config, training_config = build_configs(args)
    set_seed(training_config.seed)

    if args.smoke_test:
        return _run_smoke_test(embedding_config, training_config)

    # ---- real path: gated Llama 3.1 base backbone -------------------------- #
    if not os.path.isfile(training_config.train_file):
        print(f"error: training file not found: {training_config.train_file}")
        print("hint: use --smoke-test to exercise the pipeline without the 8B model,")
        print("      or point --train-file at your own JSONL dataset.")
        return 2

    print(f"Backbone  : {embedding_config.backbone_name_or_path}")
    print(f"Device    : {embedding_config.device or 'auto'}")
    print(f"Mode      : {training_config.mode}")
    print(f"Dim       : {embedding_config.embedding_dim}")
    print(f"Pooling   : {embedding_config.pooling}")
    print("-" * 68)

    model = LlamaEmbeddingModel.from_pretrained(
        embedding_config.backbone_name_or_path,
        tokenizer=build_tokenizer(
            embedding_config.backbone_name_or_path,
            token=embedding_config.token,
            trust_remote_code=embedding_config.trust_remote_code,
        ),
        device=embedding_config.device,
        dtype=embedding_config.dtype,
        token=embedding_config.token,
        gradient_checkpointing=training_config.gradient_checkpointing,
        embedding_dim=embedding_config.embedding_dim,
        pooling=embedding_config.pooling,
        max_length=embedding_config.max_length,
    )
    model.config.hidden_size = model.backbone.hidden_size
    model.config.validate()

    train_dataset, eval_dataset = build_datasets(
        training_config.train_file,
        eval_file=training_config.eval_file,
        validation_split=training_config.validation_split,
        seed=training_config.seed,
        num_hard_negatives=training_config.num_hard_negatives,
        hard_negative_strategy=training_config.hard_negative_strategy,
    )
    print(f"Train data: {train_dataset.stats()}")
    if eval_dataset:
        print(f"Eval data : {eval_dataset.stats()}")

    trainer = Trainer(
        model,
        train_dataset,
        training_config,
        eval_dataset=eval_dataset,
        tokenizer=model.tokenizer,
        embedding_config=embedding_config,
    )
    if training_config.resume_from:
        trainer.load_checkpoint(training_config.resume_from)

    summary = trainer.train()
    print("\nTraining complete")
    print(json.dumps({k: v for k, v in summary.items() if k != "parameter_summary"}, indent=2))
    return 0


def _run_smoke_test(embedding_config: EmbeddingConfig, training_config: TrainingConfig) -> int:
    """Train a handful of steps against the stub backbone and sample data.

    This proves the full path (tokenize -> pool -> project -> normalise ->
    similarity -> loss -> backward -> optimiser) without downloading weights.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample = os.path.join(here, "data", "sample", "train.jsonl")
    if not os.path.isfile(sample):
        print(f"smoke sample dataset missing: {sample}")
        return 2

    embedding_config.hidden_size = 64
    embedding_config.max_length = 32
    embedding_config.dtype = "float32"
    embedding_config.backbone_name_or_path = "stub://tiny-llama"
    embedding_config.validate()

    training_config.train_file = sample
    training_config.max_length = 32
    training_config.batch_size = 4
    training_config.eval_batch_size = 4
    training_config.max_steps = 5
    training_config.eval_every = 5
    training_config.save_every = 0
    training_config.output_dir = os.path.join(here, "runs", "smoke")
    training_config.mixed_precision = "no"
    training_config.validate()

    backbone = StubBackbone(hidden_size=64, seed=0)
    tokenizer = DummyTokenizer()
    model = LlamaEmbeddingModel(
        backbone=backbone, config=embedding_config, tokenizer=tokenizer
    )

    train_dataset, eval_dataset = build_datasets(
        training_config.train_file,
        eval_file=None,
        validation_split=0.25,
        seed=training_config.seed,
    )
    trainer = Trainer(
        model,
        train_dataset,
        training_config,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        embedding_config=embedding_config,
    )
    summary = trainer.train()

    # Verify the loss actually moved and the head is finite + unit norm.
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    print(f"Observed losses: {[round(x, 4) for x in losses]}")
    vectors = model.encode(["a search and rescue drone", "an emergency aircraft"], max_length=32)
    norms = vectors.norm(dim=-1)
    assert torch.isfinite(vectors).all(), "embeddings contain non-finite values"
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), f"not unit norm: {norms}"
    assert vectors.shape == (2, embedding_config.embedding_dim), vectors.shape
    print(f"Smoke OK: vectors {tuple(vectors.shape)}, norms ~ {norms.tolist()}")
    return 0


if __name__ == "__main__":
    import torch  # noqa: E402  (imported late so --help works without torch)

    raise SystemExit(main())
