"""Tests that run **without PyTorch**.

Config parsing/validation and dataset loading have no tensor dependency, so
they are verifiable on any machine — including CI images and phones where
installing torch is impractical. These tests are not skipped when torch is
missing.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

from tests.conftest import REPO_ROOT


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_yaml_parser_handles_nested_mappings_and_lists(tmp_path):
    from llama_embedding.config import load_yaml

    path = tmp_path / "c.yaml"
    path.write_text(
        """
# a comment
embedding:
  embedding_dim: 768   # trailing comment
  pooling: "mean"
  matryoshka_dims: [1024, 768, 512]
  device: null
  prompts:
    query: "query: "
training:
  mode: lora
  lora_target_modules: ["q_proj", "v_proj"]
""",
        encoding="utf-8",
    )
    data = load_yaml(str(path))
    assert data["embedding"]["embedding_dim"] == 768
    assert data["embedding"]["pooling"] == "mean"
    assert data["embedding"]["matryoshka_dims"] == [1024, 768, 512]
    assert data["embedding"]["device"] is None
    assert data["embedding"]["prompts"]["query"] == "query: "
    assert data["training"]["mode"] == "lora"
    assert data["training"]["lora_target_modules"] == ["q_proj", "v_proj"]


def test_yaml_parser_handles_block_lists():
    from llama_embedding.config import load_yaml

    path = os.path.join(REPO_ROOT, "configs", "lora.yaml")
    data = load_yaml(path)
    assert data["training"]["lora_target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]


@pytest.mark.parametrize(
    "name,expected_dim,expected_mode",
    [
        ("base.yaml", 768, "projection_only"),
        ("projection.yaml", 768, "projection_only"),
        ("lora.yaml", 768, "lora"),
    ],
)
def test_shipped_configs_are_valid(name, expected_dim, expected_mode):
    from llama_embedding.config import EmbeddingConfig, TrainingConfig, load_yaml

    data = load_yaml(os.path.join(REPO_ROOT, "configs", name))
    emb = EmbeddingConfig.from_dict(data["embedding"])
    train = TrainingConfig.from_dict(data["training"])
    assert emb.embedding_dim == expected_dim
    assert train.mode == expected_mode


def test_config_rejects_unsupported_embedding_dim():
    from llama_embedding.config import EmbeddingConfig

    with pytest.raises(ValueError, match="embedding_dim must be one of"):
        EmbeddingConfig(embedding_dim=7)


def test_config_rejects_unsupported_pooling():
    from llama_embedding.config import EmbeddingConfig

    with pytest.raises(ValueError, match="pooling must be one of"):
        EmbeddingConfig(pooling="cls_token")


def test_config_rejects_unsupported_mode():
    from llama_embedding.config import TrainingConfig

    with pytest.raises(ValueError, match="mode must be one of"):
        TrainingConfig(mode="everything", train_file="x.jsonl")


def test_config_rejects_non_positive_temperature():
    from llama_embedding.config import TrainingConfig

    with pytest.raises(ValueError, match="temperature must be > 0"):
        TrainingConfig(temperature=0.0, train_file="x.jsonl")


def test_config_rejects_implausible_temperature():
    from llama_embedding.config import TrainingConfig

    with pytest.raises(ValueError, match="implausibly small"):
        TrainingConfig(temperature=1e-9, train_file="x.jsonl")


def test_config_roundtrips_through_dict():
    from llama_embedding.config import EmbeddingConfig

    original = EmbeddingConfig(embedding_dim=1024, pooling="last_token", hidden_size=4096)
    restored = EmbeddingConfig.from_dict(original.to_dict())
    assert restored.embedding_dim == 1024
    assert restored.pooling == "last_token"


def test_config_to_dict_strips_token():
    from llama_embedding.config import EmbeddingConfig

    data = EmbeddingConfig(token="hf_do_not_leak").to_dict()
    assert data["token"] is None


def test_config_saved_file_contains_no_secret(tmp_path):
    from llama_embedding.config import EmbeddingConfig

    path = str(tmp_path / "cfg.json")
    EmbeddingConfig(token="hf_do_not_leak").save(path)
    with open(path, "r", encoding="utf-8") as handle:
        assert "hf_do_not_leak" not in handle.read()


def test_default_temperature_is_the_documented_value():
    from llama_embedding.config import DEFAULT_TEMPERATURE

    assert DEFAULT_TEMPERATURE == 0.02


def test_supported_dims_are_the_documented_set():
    from llama_embedding.config import SUPPORTED_EMBEDDING_DIMS, SUPPORTED_MODES, SUPPORTED_POOLING

    assert SUPPORTED_EMBEDDING_DIMS == (256, 384, 512, 768, 1024)
    assert SUPPORTED_POOLING == ("mean", "last_token", "weighted_mean")
    assert SUPPORTED_MODES == ("projection_only", "lora", "full")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def dataset_module():
    """Import ``training/dataset.py`` directly: its package __init__ needs torch."""
    spec = importlib.util.spec_from_file_location(
        "dataset_no_torch", os.path.join(REPO_ROOT, "training", "dataset.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["dataset_no_torch"] = module
    spec.loader.exec_module(module)
    return module


def test_loads_sample_training_rows(dataset_module):
    examples, skipped = dataset_module.load_jsonl(
        os.path.join(REPO_ROOT, "data", "sample", "train.jsonl")
    )
    assert skipped == 0
    assert len(examples) == 8
    assert all(e.query and e.positive for e in examples)


def test_accepts_both_query_and_anchor_keys(dataset_module, tmp_path):
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"query": "q1", "positive": "p1"}),
                json.dumps({"anchor": "a1", "positive": "p1"}),
                json.dumps({"query": "q2", "positive": "p2", "negative": "n1"}),
                json.dumps({"query": "q3", "positive": "p2", "hard_negatives": ["n2", "n3"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    examples, skipped = dataset_module.load_jsonl(str(path))
    assert skipped == 0
    assert len(examples) == 4
    assert examples[1].query == "a1"
    assert examples[2].negatives == ["n1"]
    assert examples[3].negatives == ["n2", "n3"]


def test_skips_malformed_rows(dataset_module, tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"query": "a", "positive": "b"}),
                "{not json",
                json.dumps({"query": "no positive"}),
                "",
                json.dumps(["not", "an", "object"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    examples, skipped = dataset_module.load_jsonl(str(path))
    assert len(examples) == 1
    assert skipped == 3


def test_split_is_reproducible(dataset_module):
    path = os.path.join(REPO_ROOT, "data", "sample", "train.jsonl")
    a_train, a_val = dataset_module.build_datasets(path, validation_split=0.25, seed=42)
    b_train, b_val = dataset_module.build_datasets(path, validation_split=0.25, seed=42)
    assert len(a_val) == len(b_val) == 2
    assert [e.query for e in a_train] == [e.query for e in b_train]
    assert [e.query for e in a_val] == [e.query for e in b_val]
    assert not ({e.query for e in a_train} & {e.query for e in a_val})


def test_duplicates_are_filtered(dataset_module, tmp_path):
    path = tmp_path / "dupes.jsonl"
    rows = [{"query": "q", "positive": "p"} for _ in range(5)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    dataset, _ = dataset_module.build_datasets(str(path), validation_split=0.0)
    assert len(dataset) == 1
    assert dataset.num_duplicates_removed == 4


def test_negative_sampling_is_seeded(dataset_module):
    path = os.path.join(REPO_ROOT, "data", "sample", "train.jsonl")
    import random

    dataset, _ = dataset_module.build_datasets(
        path, validation_split=0.0, num_hard_negatives=1, hard_negative_strategy="provided"
    )
    for example in dataset:
        first = dataset.negatives_for(example, random.Random(1))
        second = dataset.negatives_for(example, random.Random(1))
        assert first == second


def test_random_negatives_never_equal_positive(dataset_module):
    import random

    path = os.path.join(REPO_ROOT, "data", "sample", "train.jsonl")
    dataset, _ = dataset_module.build_datasets(
        path, validation_split=0.0, num_hard_negatives=1, hard_negative_strategy="random"
    )
    for example in dataset:
        negatives = dataset.negatives_for(example, random.Random(7))
        assert all(n != example.positive for n in negatives)


def test_eval_sample_file_is_parseable_by_retrieval_loader(dataset_module):
    """The evaluation fixture must round-trip through the retrieval loader."""
    pytest.importorskip("torch", reason="evaluation.retrieval imports torch")
    from evaluation.retrieval import load_retrieval_jsonl

    corpus, examples = load_retrieval_jsonl(
        os.path.join(REPO_ROOT, "data", "sample", "eval.jsonl")
    )
    assert len(corpus) == 10
    assert len(examples) == 7
    assert all(e.relevant_ids for e in examples)
    assert len({doc_id for e in examples for doc_id in e.relevant_ids}) > 1
