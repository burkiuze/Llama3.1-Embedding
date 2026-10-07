"""Dataset / collator tests: parsing, dedupe, seeding, batching, collation."""

from __future__ import annotations

import json
import random

import pytest

torch = pytest.importorskip("torch")

from llama_embedding.config import PromptTemplates
from llama_embedding.tokenizer import DummyTokenizer
from training.collator import ContrastiveCollator
from training.dataset import (
    ContrastiveDataset,
    ContrastiveExample,
    build_datasets,
    dedupe_examples,
    load_jsonl,
)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


@pytest.fixture
def triplet_file(tmp_path):
    path = tmp_path / "triplet.jsonl"
    rows = [
        {"query": "how tall is everest", "positive": "everest is 8849m", "negative": "everest is a mountain"},
        {"query": "capital of france", "positive": "paris is the capital", "negative": "lyon is in france"},
        {"anchor": "python list append", "positive": "list.append(x) adds an item"},
    ]
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return str(path)


@pytest.fixture
def negatives_list_file(tmp_path):
    path = tmp_path / "negs.jsonl"
    rows = [
        {
            "query": "q1",
            "positive": "p1",
            "hard_negatives": ["n1", "n2"],
        },
        {"query": "q2", "positive": "p2", "negatives": ["n3"]},
    ]
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return str(path)


def test_loads_triplet_format(triplet_file):
    examples, skipped = load_jsonl(triplet_file)
    assert len(examples) == 3
    assert skipped == 0
    assert examples[0].query == "how tall is everest"
    assert examples[0].positive == "everest is 8849m"
    assert examples[0].negatives == ["everest is a mountain"]


def test_accepts_anchor_key(triplet_file):
    examples, _ = load_jsonl(triplet_file)
    assert examples[2].query == "python list append"


def test_loads_hard_negative_lists(negatives_list_file):
    examples, _ = load_jsonl(negatives_list_file)
    assert examples[0].negatives == ["n1", "n2"]
    assert examples[1].negatives == ["n3"]


def test_skips_malformed_rows(tmp_path):
    path = tmp_path / "bad.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write('{"query": "a", "positive": "b"}\n')
        handle.write("not json at all\n")
        handle.write('{"query": "missing positive"}\n')
        handle.write('{"positive": "missing query"}\n')
    examples, skipped = load_jsonl(str(path))
    assert len(examples) == 1
    assert skipped == 3


def test_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_jsonl("/nonexistent/path.jsonl")


def test_max_rows_limits_output(triplet_file):
    examples, _ = load_jsonl(triplet_file, max_rows=2)
    assert len(examples) == 2


# --------------------------------------------------------------------------- #
# Deduplication
# --------------------------------------------------------------------------- #


def test_dedupe_removes_exact_duplicates():
    examples = [
        ContrastiveExample("a", "b"),
        ContrastiveExample("a", "b"),
        ContrastiveExample("a", "c"),
    ]
    assert len(dedupe_examples(examples)) == 2


def test_dataset_reports_duplicate_count():
    examples = [ContrastiveExample("a", "b")] * 3 + [ContrastiveExample("x", "y")]
    dataset = ContrastiveDataset(examples, dedupe=True, shuffle=False)
    assert len(dataset) == 2
    assert dataset.num_duplicates_removed == 2


# --------------------------------------------------------------------------- #
# Determinism / seeding
# --------------------------------------------------------------------------- #


def test_same_seed_same_order(triplet_file):
    a, _ = build_datasets(triplet_file, validation_split=0.0, seed=7)
    b, _ = build_datasets(triplet_file, validation_split=0.0, seed=7)
    assert [e.query for e in a] == [e.query for e in b]


def test_different_seed_different_order(triplet_file):
    """With 3 rows, two distinct seeds must not always produce the same order.

    Rather than assume a specific permutation differs, assert that across a few
    seed pairs at least one differs (a 3! search over 2 samples is deterministic).
    """
    orders = []
    for seed in range(1, 8):
        dataset, _ = build_datasets(triplet_file, validation_split=0.0, seed=seed)
        orders.append(tuple(e.query for e in dataset))
    assert len(set(orders)) > 1, "different seeds produced identical orders"


def test_shuffle_false_preserves_file_order(triplet_file):
    dataset, _ = build_datasets(triplet_file, validation_split=0.0, shuffle=False)
    assert dataset[0].query == "how tall is everest"


def test_validation_split_is_deterministic_and_disjoint(triplet_file):
    train_a, val_a = build_datasets(triplet_file, validation_split=0.34, seed=5)
    train_b, val_b = build_datasets(triplet_file, validation_split=0.34, seed=5)
    assert len(val_a) == len(val_b) == 1
    assert [e.query for e in train_a] == [e.query for e in train_b]
    assert [e.query for e in val_a] == [e.query for e in val_b]
    assert not ({e.query for e in train_a} & {e.query for e in val_a})


def test_no_validation_split_returns_none(triplet_file):
    _, val = build_datasets(triplet_file, validation_split=0.0)
    assert val is None


def test_hard_negative_sampling_is_seeded(negatives_list_file):
    dataset, _ = build_datasets(
        negatives_list_file, validation_split=0.0, seed=3,
        num_hard_negatives=2, hard_negative_strategy="provided",
    )
    rng1 = random.Random(3)
    rng2 = random.Random(3)
    for example in dataset:
        assert dataset.negatives_for(example, rng1) == dataset.negatives_for(example, rng2)


def test_random_negative_strategy_avoids_positive(negatives_list_file):
    dataset, _ = build_datasets(
        negatives_list_file, validation_split=0.0, seed=9,
        num_hard_negatives=1, hard_negative_strategy="random",
    )
    rng = random.Random(9)
    for example in dataset:
        negatives = dataset.negatives_for(example, rng)
        assert all(n != example.positive for n in negatives)


def test_num_hard_negatives_zero_returns_empty(negatives_list_file):
    dataset, _ = build_datasets(negatives_list_file, validation_split=0.0, num_hard_negatives=0)
    rng = random.Random(0)
    for example in dataset:
        assert dataset.negatives_for(example, rng) == []


def test_get_batch_shapes(negatives_list_file):
    dataset, _ = build_datasets(negatives_list_file, validation_split=0.0, num_hard_negatives=2)
    batch = dataset.get_batch([0, 1])
    assert len(batch["queries"]) == 2
    assert len(batch["positives"]) == 2
    assert len(batch["negatives"]) == 2


def test_stats_dict(triplet_file):
    dataset, _ = build_datasets(triplet_file, validation_split=0.0)
    stats = dataset.stats()
    assert stats["num_examples"] == 3
    assert stats["examples_with_provided_negatives"] == 2


# --------------------------------------------------------------------------- #
# Collator
# --------------------------------------------------------------------------- #


def test_collator_produces_padded_tensors():
    tokenizer = DummyTokenizer(vocab_size=1024)
    collator = ContrastiveCollator(tokenizer=tokenizer, max_length=16)
    features = [
        {"queries": "a short one", "positives": "a much longer positive passage here", "negatives": []},
        {"queries": "another query", "positives": "second positive", "negatives": []},
    ]
    batch = collator(features)
    q = batch["query"]
    assert q["input_ids"].shape == q["attention_mask"].shape
    assert q["input_ids"].shape[0] == 2
    assert batch["positive"]["input_ids"].shape[0] == 2
    assert batch["num_negatives"] == 0


def test_collator_pads_with_mask_zeros():
    """Right padding: the shorter query gets mask=0 in the extra columns."""
    tokenizer = DummyTokenizer(vocab_size=1024)
    collator = ContrastiveCollator(tokenizer=tokenizer, max_length=16)
    features = [
        {"queries": "one two three four", "positives": "one two", "negatives": []},
        {"queries": "a", "positives": "b c d e f", "negatives": []},
    ]
    batch = collator(features)
    mask = batch["query"]["attention_mask"]
    assert mask.shape == (2, 4)
    assert mask[0].tolist() == [1, 1, 1, 1]   # 4 real tokens, no padding
    assert mask[1].tolist() == [1, 0, 0, 0]   # 1 real token + 3 padding
    # Padded positions must not carry a real token id.
    assert mask[1].tolist()[1:] == [0, 0, 0]


def test_collator_mask_marks_exactly_the_real_tokens():
    """Independent of padding width: mask sum == number of whitespace words."""
    tokenizer = DummyTokenizer(vocab_size=1024)
    collator = ContrastiveCollator(tokenizer=tokenizer, max_length=16)
    features = [
        {"queries": "one two three four", "positives": "a b", "negatives": []},
        {"queries": "a", "positives": "b c d e f", "negatives": []},
    ]
    batch = collator(features)
    qmask = batch["query"]["attention_mask"]
    pmask = batch["positive"]["attention_mask"]
    assert int(qmask[0].sum()) == 4 and int(qmask[1].sum()) == 1
    assert int(pmask[0].sum()) == 2 and int(pmask[1].sum()) == 5


def test_collator_applies_prompt_templates():
    """The template prefix adds one token on each side (DummyTokenizer splits on space)."""
    tokenizer = DummyTokenizer(vocab_size=1024)
    prompts = PromptTemplates(query="Q:", document="D:")
    collator = ContrastiveCollator(tokenizer=tokenizer, max_length=16, prompts=prompts)
    features = [{"queries": "hello", "positives": "world", "negatives": []}]
    batch = collator(features)
    # "hello" -> 1 token; with the "Q:" prefix -> 2.
    assert int(batch["query"]["attention_mask"][0].sum()) == 2
    assert int(batch["positive"]["attention_mask"][0].sum()) == 2

    # And the prefix really is in the text: a longer prefix adds proportionally.
    longer = ContrastiveCollator(
        tokenizer=tokenizer, max_length=16,
        prompts=PromptTemplates(query="Q Q:", document="D D:"),
    )
    batch2 = longer(features)
    assert int(batch2["query"]["attention_mask"][0].sum()) == 3
    assert int(batch2["positive"]["attention_mask"][0].sum()) == 3


def test_collator_with_negatives_shapes():
    tokenizer = DummyTokenizer(vocab_size=1024)
    collator = ContrastiveCollator(tokenizer=tokenizer, max_length=16)
    features = [
        {"queries": "q one", "positives": "p one", "negatives": ["n a", "n b"]},
        {"queries": "q two", "positives": "p two", "negatives": ["n c", "n d"]},
    ]
    batch = collator(features)
    assert batch["num_negatives"] == 2
    neg = batch["negative"]
    assert neg["input_ids"].shape[0] == 2      # batch
    assert neg["input_ids"].shape[1] == 2      # negatives per item
    assert neg["input_ids"].shape[2] == neg["attention_mask"].shape[2]


def test_collator_negative_tensor_ordering_is_item_major():
    """Regression: [B, K, T] must be item-major, valid even when K != B.

    The previous implementation used ``view(K, B, T).permute(1, 0, 2)``, which
    only happens to be correct when K == B. With B=3, K=2 it silently paired
    every query with the wrong negatives.
    """
    tokenizer = DummyTokenizer(vocab_size=1024)
    collator = ContrastiveCollator(tokenizer=tokenizer, max_length=8)
    features = [
        {"queries": "qA", "positives": "pA", "negatives": ["nA0", "nA1"]},
        {"queries": "qB", "positives": "pB", "negatives": ["nB0", "nB1"]},
        {"queries": "qC", "positives": "pC", "negatives": ["nC0", "nC1"]},
    ]
    batch = collator(features)
    neg_ids = batch["negative"]["input_ids"]
    neg_mask = batch["negative"]["attention_mask"]

    batch_size, num_negatives, _ = neg_ids.shape
    assert batch_size == 3 and num_negatives == 2

    # Distinct single-token negatives make the ordering observable.
    for b, feature in enumerate(features):
        for k, expected in enumerate(feature["negatives"]):
            encoded = tokenizer([expected], max_length=8)["input_ids"][0]
            assert neg_ids[b, k].tolist() == encoded
            assert int(neg_mask[b, k].sum()) == len(encoded)


def test_collator_pads_short_negative_lists_by_repetition():
    """A ragged K must not inject empty-string rows into the loss."""
    tokenizer = DummyTokenizer(vocab_size=1024)
    collator = ContrastiveCollator(tokenizer=tokenizer, max_length=8)
    features = [
        {"queries": "qA", "positives": "pA", "negatives": ["onlyOne"]},
        {"queries": "qB", "positives": "pB", "negatives": ["nB0", "nB1"]},
    ]
    batch = collator(features)
    assert batch["num_negatives"] == 2
    neg_mask = batch["negative"]["attention_mask"]
    # Row 0's two negatives are identical (the repeat), and both are non-empty.
    assert torch.equal(neg_mask[0, 0], neg_mask[0, 1])
    assert int(neg_mask[0, 0].sum()) > 0
    assert not torch.equal(neg_mask[0, 0], neg_mask[1, 0])


def test_collator_rejects_empty_batch():
    collator = ContrastiveCollator(tokenizer=DummyTokenizer(), max_length=8)
    with pytest.raises(ValueError, match="empty batch"):
        collator([])


def test_collator_rejects_bad_features():
    collator = ContrastiveCollator(tokenizer=DummyTokenizer(), max_length=8)
    with pytest.raises(ValueError, match="'queries' and 'positives'"):
        collator([{"text": "oops"}])


def test_collator_rejects_missing_tokenizer():
    with pytest.raises(ValueError, match="tokenizer is required"):
        ContrastiveCollator(tokenizer=None, max_length=8)


def test_collator_respects_max_length():
    tokenizer = DummyTokenizer(vocab_size=1024)
    collator = ContrastiveCollator(tokenizer=tokenizer, max_length=4)
    features = [{"queries": " ".join(["w"] * 20), "positives": "x", "negatives": []}]
    batch = collator(features)
    assert batch["query"]["input_ids"].shape[1] <= 4


# --------------------------------------------------------------------------- #
# Build integration
# --------------------------------------------------------------------------- #


def test_build_datasets_from_empty_file_raises(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no usable examples"):
        build_datasets(str(path))


def test_eval_file_overrides_split(negatives_list_file):
    _, val = build_datasets(negatives_list_file, eval_file=negatives_list_file, validation_split=0.5)
    assert val is not None and val.source == negatives_list_file
    assert val.shuffle is False
