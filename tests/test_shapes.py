"""End-to-end shape, batching and determinism tests for the encoder.

These use the :class:`StubBackbone`, so they validate the *full* pipeline
(tokenizer -> backbone -> pooling -> projection -> normalisation) without ever
touching the gated 8B weights.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from llama_embedding.config import EmbeddingConfig
from llama_embedding.model import LlamaEmbeddingModel, StubBackbone
from llama_embedding.tokenizer import DummyTokenizer


# --------------------------------------------------------------------------- #
# Forward shapes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dim", [256, 384, 512, 768, 1024])
def test_encode_returns_num_texts_by_dim(tiny_model, stub_tokenizer, dim):
    tiny_model.head = tiny_model.head.__class__(
        hidden_size=tiny_model.backbone.hidden_size, embedding_dim=dim
    )
    tiny_model.config.embedding_dim = dim
    vectors = tiny_model.encode(["a", "b", "c"])
    assert vectors.shape == (3, dim)


def test_forward_output_shape(tiny_model):
    input_ids = torch.randint(0, 2000, (4, 16))
    mask = torch.ones_like(input_ids)
    out = tiny_model(input_ids, mask)
    assert out["embeddings"].shape == (4, tiny_model.config.embedding_dim)


def test_forward_can_return_pooled_and_token_embeddings(tiny_model):
    input_ids = torch.randint(0, 2000, (2, 10))
    mask = torch.ones_like(input_ids)
    out = tiny_model(input_ids, mask, return_pooled=True, return_token_embeddings=True)
    assert out["pooled"].shape == (2, tiny_model.backbone.hidden_size)
    assert out["token_embeddings"].shape == (2, 10, tiny_model.backbone.hidden_size)


def test_embeddings_are_unit_norm_after_encode(tiny_model):
    vectors = tiny_model.encode(["one", "two", "three"])
    assert torch.allclose(vectors.norm(dim=-1), torch.ones(3), atol=1e-4)


# --------------------------------------------------------------------------- #
# Public API: single string, list, batching
# --------------------------------------------------------------------------- #


def test_encode_single_string_returns_2d(tiny_model):
    vectors = tiny_model.encode("a single string")
    assert vectors.ndim == 2
    assert vectors.shape[0] == 1


def test_encode_empty_list_returns_empty_tensor(tiny_model):
    out = tiny_model.encode([])
    assert out.shape == (0, tiny_model.config.embedding_dim)


def test_encode_batching_is_consistent(tiny_model):
    texts = [f"text number {i}" for i in range(7)]
    one_shot = tiny_model.encode(texts, batch_size=7)
    batched = tiny_model.encode(texts, batch_size=2)
    assert torch.allclose(one_shot, batched, atol=1e-5)


def test_encode_respects_max_length(tiny_model):
    long_text = " ".join(["word"] * 500)
    vectors = tiny_model.encode([long_text], max_length=16)
    assert vectors.shape == (1, tiny_model.config.embedding_dim)


def test_encode_rejects_non_string(tiny_model):
    with pytest.raises(TypeError, match="string or a sequence of strings"):
        tiny_model.encode([123, 456])


def test_encode_requires_tokenizer(tiny_config):
    model = LlamaEmbeddingModel(
        backbone=StubBackbone(hidden_size=64, seed=0), config=tiny_config, tokenizer=None
    )
    with pytest.raises(RuntimeError, match="no tokenizer"):
        model.encode(["x"])


# --------------------------------------------------------------------------- #
# Query / document modes
# --------------------------------------------------------------------------- #


def test_query_and_document_modes_share_architecture(tiny_config):
    tiny_config.prompts.query = "query: "
    tiny_config.prompts.document = "passage: "
    model = LlamaEmbeddingModel(
        backbone=StubBackbone(hidden_size=64, seed=0),
        config=tiny_config,
        tokenizer=DummyTokenizer(),
    )
    q = model.encode_query("search and rescue")
    d = model.encode_document("search and rescue")
    assert q.shape == d.shape == (1, tiny_config.embedding_dim)
    # Different templates should give different vectors for the same text.
    assert not torch.allclose(q, d, atol=1e-3)


def test_prompt_templates_default_to_no_prefix(tiny_config):
    assert tiny_config.prompts.apply("hello", "query") == "hello"
    assert tiny_config.prompts.apply("hello", "document") == "hello"


def test_prompt_apply_rejects_bad_role(tiny_config):
    with pytest.raises(ValueError, match="role must be"):
        tiny_config.prompts.apply("hello", "passage")


def test_templates_are_language_agnostic(tiny_config):
    """Any language may be configured without code changes."""
    tiny_config.prompts.query = "Suchfrage: "
    tiny_config.prompts.document = "Dokument: "
    assert tiny_config.prompts.apply("Frage", "query") == "Suchfrage: Frage"
    assert tiny_config.prompts.apply("Dok", "document") == "Dokument: Dok"


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_encode_is_deterministic(tiny_model):
    texts = ["alpha", "beta", "gamma"]
    a = tiny_model.encode(texts)
    b = tiny_model.encode(texts)
    assert torch.allclose(a, b, atol=1e-6)


def test_same_seed_gives_same_model_outputs():
    def build():
        torch.manual_seed(42)
        config = EmbeddingConfig(backbone_name_or_path="stub", hidden_size=32,
                                 embedding_dim=256, max_length=16, dtype="float32")
        model = LlamaEmbeddingModel(
            backbone=StubBackbone(vocab_size=1024, hidden_size=32, seed=42),
            config=config,
            tokenizer=DummyTokenizer(vocab_size=1024),
        )
        return model.encode(["hello world", "second text"], max_length=16)

    assert torch.allclose(build(), build(), atol=1e-6)


def test_similarity_helper_returns_matrix(tiny_model):
    sim = tiny_model.similarity(["a", "b"], ["c", "d"])
    assert sim.shape == (2, 2)


# --------------------------------------------------------------------------- #
# Trainable modes
# --------------------------------------------------------------------------- #


def test_projection_only_freezes_backbone(tiny_model):
    info = tiny_model.set_trainable_mode("projection_only")
    assert all(not p.requires_grad for p in tiny_model.backbone.parameters())
    assert all(p.requires_grad for p in tiny_model.head.parameters())
    assert 0 < info["trainable_percent"] < 50


def test_full_mode_unfreezes_everything(tiny_model):
    info = tiny_model.set_trainable_mode("full")
    assert all(p.requires_grad for p in tiny_model.backbone.parameters())
    assert info["trainable_percent"] == pytest.approx(100.0, abs=1e-6)


def test_unknown_mode_raises(tiny_model):
    with pytest.raises(ValueError, match="unknown training mode"):
        tiny_model.set_trainable_mode("magic")


def test_head_state_dict_roundtrip(tmp_path, tiny_model):
    """Saving and reloading the head must reproduce identical weights."""
    import json as _json

    tiny_model.set_trainable_mode("projection_only")
    out_dir = str(tmp_path / "ckpt")
    tiny_model.save_pretrained(out_dir)

    assert os.path.isfile(os.path.join(out_dir, "embedding_head.pt"))
    assert os.path.isfile(os.path.join(out_dir, "embedding_config.json"))
    assert os.path.isfile(os.path.join(out_dir, "metadata.json"))

    state = torch.load(os.path.join(out_dir, "embedding_head.pt"), weights_only=True)
    assert all(k.startswith("head.") for k in state)

    fresh = LlamaEmbeddingModel(
        backbone=StubBackbone(hidden_size=tiny_model.backbone.hidden_size, seed=99),
        config=EmbeddingConfig.from_file(os.path.join(out_dir, "embedding_config.json")),
        tokenizer=DummyTokenizer(),
    )
    fresh.load_head(state)
    pooled = torch.randn(3, tiny_model.backbone.hidden_size)
    assert torch.allclose(fresh.head(pooled), tiny_model.head(pooled), atol=1e-6)

    with open(os.path.join(out_dir, "metadata.json"), "r", encoding="utf-8") as handle:
        meta = _json.load(handle)
    assert meta["embedding_dim"] == tiny_model.config.embedding_dim


def test_saved_config_never_contains_token(tmp_path):
    config = EmbeddingConfig(token="hf_SUPERSECRET", hidden_size=64)
    out = str(tmp_path / "c.json")
    config.save(out)
    with open(out, "r", encoding="utf-8") as handle:
        assert "hf_SUPERSECRET" not in handle.read()


# --------------------------------------------------------------------------- #
# Invalid dimension handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_dim", [0, -1, 100, 769, 2048])
def test_invalid_dimensions_are_rejected(bad_dim):
    with pytest.raises(ValueError):
        EmbeddingConfig(hidden_size=4096, embedding_dim=bad_dim)


def test_forward_rejects_bad_input_rank(tiny_model):
    with pytest.raises(ValueError, match=r"\[B, T\]"):
        tiny_model(torch.randn(2, 3, 4), torch.ones(2, 3, dtype=torch.long))


def test_forward_rejects_mask_shape_mismatch(tiny_model):
    with pytest.raises(ValueError, match="attention_mask shape"):
        tiny_model(torch.zeros(2, 5, dtype=torch.long), torch.ones(2, 4, dtype=torch.long))
