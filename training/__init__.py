"""Training pipeline for Llama3.1-Embedding.

Exports the dataset, collator and trainer so external code can build them
without touching the CLI.
"""

from __future__ import annotations

from .collator import ContrastiveCollator, EvalCollator, encode_text_group
from .dataset import (
    ContrastiveDataset,
    ContrastiveExample,
    build_datasets,
    dedupe_examples,
    load_jsonl,
)
from .trainer import NonFiniteLossError, Trainer, TrainerState, set_seed

__all__ = [
    "ContrastiveExample",
    "ContrastiveDataset",
    "load_jsonl",
    "build_datasets",
    "dedupe_examples",
    "ContrastiveCollator",
    "EvalCollator",
    "encode_text_group",
    "Trainer",
    "TrainerState",
    "NonFiniteLossError",
    "set_seed",
]
