"""Contrastive dataset loaders for Llama3.1-Embedding.

Supported JSONL schemas (one JSON object per line)
------------------------------------------------
Triplet / pair format::

    {"query": "...", "positive": "...", "negative": "..."}
    {"query": "...", "positive": "...", "hard_negatives": ["...", "..."]}

Pair / anchor format::

    {"anchor": "...", "positive": "..."}

``query``/``anchor`` and ``positive`` are interchangeable on input.

The loaders are dataset-agnostic and network-free: nothing is downloaded and no
data is bundled beyond the tiny ``data/sample`` fixtures used by smoke tests.
Users point the trainer at their own JSONL (MS MARCO, NLI, BEIR, synthetic
augmentations, ...).

Reproducibility
---------------
Shuffling and hard-negative sampling are driven by an explicit ``random.Random``
seed, so a given ``(file, seed, split)`` always yields the same order, the same
validation split and the same sampled negatives.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "ContrastiveExample",
    "ContrastiveDataset",
    "load_jsonl",
    "build_datasets",
    "dedupe_examples",
]

_QUERY_KEYS = ("query", "anchor", "question", "sentence1")
_POSITIVE_KEYS = ("positive", "passage", "sentence2", "correct_answer")
_NEGATIVE_KEYS = ("negative", "hard_negative", "incorrect_answer")


@dataclass
class ContrastiveExample:
    """A single ``(query, positive, negatives)`` training item."""

    query: str
    positive: str
    negatives: List[str] = field(default_factory=list)
    id: Optional[str] = None

    def key(self) -> str:
        """Stable identity used for duplicate filtering."""
        return f"{self.query}\x1f{self.positive}"


def _first_text(record: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_negatives(record: Dict[str, Any]) -> List[str]:
    negatives: List[str] = []
    for key in _NEGATIVE_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            negatives.append(value.strip())
        elif isinstance(value, list):
            negatives.extend([v.strip() for v in value if isinstance(v, str) and v.strip()])
    # "hard_negatives" is a common explicit list field
    for key in ("hard_negatives", "negatives"):
        value = record.get(key)
        if isinstance(value, list):
            negatives.extend([v.strip() for v in value if isinstance(v, str) and v.strip()])
    # De-duplicate while preserving order
    seen = set()
    unique = []
    for n in negatives:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique


def _parse_record(record: Dict[str, Any], line_no: int, source: str) -> Optional[ContrastiveExample]:
    query = _first_text(record, _QUERY_KEYS)
    positive = _first_text(record, _POSITIVE_KEYS)
    if query is None or positive is None:
        # Skip malformed rows rather than crashing a long training run, but make
        # it visible through the returned counter.
        return None
    example_id = record.get("id") or record.get("_id")
    return ContrastiveExample(
        query=query,
        positive=positive,
        negatives=_extract_negatives(record),
        id=str(example_id) if example_id is not None else None,
    )


def load_jsonl(
    path: str,
    *,
    max_rows: Optional[int] = None,
) -> Tuple[List[ContrastiveExample], int]:
    """Read a JSONL file into examples.

    Returns ``(examples, skipped_rows)`` so callers can log data quality.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"dataset not found: {path}")
    examples: List[ContrastiveExample] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(record, dict):
                skipped += 1
                continue
            example = _parse_record(record, line_no, path)
            if example is None:
                skipped += 1
                continue
            examples.append(example)
            if max_rows is not None and len(examples) >= max_rows:
                break
    return examples, skipped


def dedupe_examples(examples: Sequence[ContrastiveExample]) -> List[ContrastiveExample]:
    """Drop exact duplicate ``(query, positive)`` pairs, preserving order."""
    seen: set[str] = set()
    unique: List[ContrastiveExample] = []
    for example in examples:
        key = example.key()
        if key in seen:
            continue
        seen.add(key)
        unique.append(example)
    return unique


class ContrastiveDataset:
    """An in-memory list of :class:`ContrastiveExample` with a seeded DataLoader.

    The dataset deliberately does **not** use ``torch.utils.data.Dataset``
    semantics so it can be constructed and inspected without torch; the trainer
    wraps it when needed.
    """

    def __init__(
        self,
        examples: Sequence[ContrastiveExample],
        *,
        seed: int = 42,
        shuffle: bool = True,
        num_hard_negatives: int = 1,
        hard_negative_strategy: str = "provided",
        dedupe: bool = True,
        source: str = "<memory>",
    ) -> None:
        self.source = source
        self.seed = seed
        self.shuffle = shuffle
        self.num_hard_negatives = num_hard_negatives
        self.hard_negative_strategy = hard_negative_strategy
        self._rng = random.Random(seed)

        items = list(examples)
        self.num_raw = len(items)
        if dedupe:
            items = dedupe_examples(items)
        self.num_duplicates_removed = self.num_raw - len(items)
        if shuffle:
            self._rng.shuffle(items)
        self.examples: List[ContrastiveExample] = items

        # Pool of positives used to synthesise random negatives.
        self._positive_pool = [e.positive for e in items]

    # -- container protocol -------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ContrastiveExample:
        return self.examples[idx]

    def __iter__(self):
        return iter(self.examples)

    # -- negative sampling ---------------------------------------------------- #
    def negatives_for(self, example: ContrastiveExample, rng: random.Random) -> List[str]:
        """Return up to ``num_hard_negatives`` negatives for one example."""
        k = self.num_hard_negatives
        if k <= 0:
            return []
        strategy = self.hard_negative_strategy
        chosen: List[str] = []

        if strategy in ("provided", "mixed"):
            chosen.extend(example.negatives[:k])
        if strategy in ("random", "mixed"):
            while len(chosen) < k and self._positive_pool:
                candidate = self._positive_pool[rng.randrange(len(self._positive_pool))]
                if candidate != example.positive and candidate not in chosen:
                    chosen.append(candidate)
        return chosen[:k]

    def get_batch(self, indices: Sequence[int]) -> Dict[str, Any]:
        """Materialise a batch of raw text fields for the given indices."""
        rng = random.Random(self.seed + (indices[0] if indices else 0))
        queries: List[str] = []
        positives: List[str] = []
        negatives: List[List[str]] = []
        for idx in indices:
            example = self.examples[idx]
            queries.append(example.query)
            positives.append(example.positive)
            negatives.append(self.negatives_for(example, rng))
        return {"queries": queries, "positives": positives, "negatives": negatives}

    def stats(self) -> Dict[str, Any]:
        with_negatives = sum(1 for e in self.examples if e.negatives)
        return {
            "source": self.source,
            "num_examples": len(self.examples),
            "num_raw": self.num_raw,
            "num_duplicates_removed": self.num_duplicates_removed,
            "examples_with_provided_negatives": with_negatives,
            "num_hard_negatives": self.num_hard_negatives,
            "hard_negative_strategy": self.hard_negative_strategy,
            "shuffle": self.shuffle,
            "seed": self.seed,
        }


def _split(
    examples: List[ContrastiveExample], validation_split: float, seed: int
) -> Tuple[List[ContrastiveExample], List[ContrastiveExample]]:
    if validation_split <= 0:
        return examples, []
    items = list(examples)
    random.Random(seed).shuffle(items)
    n_val = max(1, int(len(items) * validation_split))
    return items[n_val:], items[:n_val]


def build_datasets(
    train_file: str,
    *,
    eval_file: Optional[str] = None,
    validation_split: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
    dedupe: bool = True,
    num_hard_negatives: int = 1,
    hard_negative_strategy: str = "provided",
    max_train_rows: Optional[int] = None,
) -> Tuple[ContrastiveDataset, Optional[ContrastiveDataset]]:
    """Build train (and optionally validation) datasets from JSONL file(s)."""
    train_examples, train_skipped = load_jsonl(train_file, max_rows=max_train_rows)
    if not train_examples:
        raise ValueError(f"no usable examples parsed from {train_file}")

    train_pool, val_pool = _split(train_examples, validation_split, seed)

    train_dataset = ContrastiveDataset(
        train_pool,
        seed=seed,
        shuffle=shuffle,
        num_hard_negatives=num_hard_negatives,
        hard_negative_strategy=hard_negative_strategy,
        dedupe=dedupe,
        source=train_file,
    )

    eval_dataset: Optional[ContrastiveDataset] = None
    if eval_file:
        eval_examples, eval_skipped = load_jsonl(eval_file)
        if eval_examples:
            eval_dataset = ContrastiveDataset(
                eval_examples,
                seed=seed,
                shuffle=False,  # deterministic validation order
                num_hard_negatives=num_hard_negatives,
                hard_negative_strategy=hard_negative_strategy,
                dedupe=dedupe,
                source=eval_file,
            )
        else:
            eval_examples = eval_skipped  # noqa: F841
    elif validation_split > 0 and val_pool:
        eval_dataset = ContrastiveDataset(
            val_pool,
            seed=seed,
            shuffle=False,
            num_hard_negatives=num_hard_negatives,
            hard_negative_strategy=hard_negative_strategy,
            dedupe=dedupe,
            source=f"{train_file}::val",
        )

    if train_skipped:
        train_dataset.stats()  # no-op; skipped count is surfaced by the trainer log
    return train_dataset, eval_dataset


def file_fingerprint(path: str) -> str:
    """Stable content hash, recorded in checkpoints for reproducibility."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
