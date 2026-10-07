"""Projection head tests: shapes, normalisation, configurability, truncation."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from llama_embedding.config import SUPPORTED_EMBEDDING_DIMS, EmbeddingConfig
from llama_embedding.projection import EmbeddingHead, count_trainable, l2_normalize


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dim", SUPPORTED_EMBEDDING_DIMS)
def test_output_shape_for_every_supported_dim(dim):
    head = EmbeddingHead(hidden_size=32, embedding_dim=dim)
    pooled = torch.randn(5, 32)
    out = head(pooled)
    assert out.shape == (5, dim)


def test_rejects_unsupported_dim():
    with pytest.raises(ValueError, match="embedding_dim must be one of"):
        EmbeddingHead(hidden_size=32, embedding_dim=999)


def test_rejects_wrong_input_rank():
    head = EmbeddingHead(hidden_size=32, embedding_dim=768)
    with pytest.raises(ValueError, match=r"\[B, hidden_size\]"):
        head(torch.randn(2, 3, 32))


def test_rejects_wrong_hidden_size():
    head = EmbeddingHead(hidden_size=32, embedding_dim=768)
    with pytest.raises(ValueError, match="last dim 32"):
        head(torch.randn(4, 64))


def test_rejects_non_positive_hidden_size():
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        EmbeddingHead(hidden_size=0, embedding_dim=768)


def test_batch_size_one_works():
    head = EmbeddingHead(hidden_size=16, embedding_dim=256)
    assert head(torch.randn(1, 16)).shape == (1, 256)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def test_output_is_unit_norm():
    """The headline invariant: ||embedding||_2 == 1."""
    head = EmbeddingHead(hidden_size=64, embedding_dim=768)
    out = head(torch.randn(16, 64))
    norms = out.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_unit_norm_without_layer_norm():
    head = EmbeddingHead(hidden_size=64, embedding_dim=512, use_layer_norm=False)
    out = head(torch.randn(8, 64) * 100)
    assert torch.allclose(out.norm(dim=-1), torch.ones(8), atol=1e-5)


def test_normalisation_can_be_disabled():
    head = EmbeddingHead(hidden_size=64, embedding_dim=768, normalize=False)
    out = head(torch.randn(8, 64))
    assert not torch.allclose(out.norm(dim=-1), torch.ones(8), atol=1e-3)


def test_l2_normalize_handles_zero_vector():
    zeros = torch.zeros(2, 4)
    out = l2_normalize(zeros)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, zeros)


def test_l2_normalize_preserves_direction():
    torch.manual_seed(0)
    x = torch.randn(10, 16)
    out = l2_normalize(x)
    cosine = (out * x).sum(-1) / (out.norm(dim=-1) * x.norm(dim=-1))
    assert torch.allclose(cosine, torch.ones(10), atol=1e-5)


def test_similar_inputs_produce_similar_unit_vectors():
    head = EmbeddingHead(hidden_size=32, embedding_dim=256)
    torch.manual_seed(3)
    base = torch.randn(1, 32)
    out = head(torch.cat([base, base * 1.01, base * 3.0], dim=0))
    sim_01 = torch.dot(out[0], out[1])
    sim_03 = torch.dot(out[0], out[2])
    assert sim_01 > 0.999
    assert sim_03 > 0.999


# --------------------------------------------------------------------------- #
# Configurability
# --------------------------------------------------------------------------- #


def test_layer_norm_is_configurable():
    with_ln = EmbeddingHead(hidden_size=32, embedding_dim=768, use_layer_norm=True)
    without_ln = EmbeddingHead(hidden_size=32, embedding_dim=768, use_layer_norm=False)
    assert isinstance(with_ln.layer_norm, torch.nn.LayerNorm)
    assert isinstance(without_ln.layer_norm, torch.nn.Identity)


def test_bias_is_configurable():
    assert EmbeddingHead(hidden_size=32, embedding_dim=768, bias=True).linear.bias is not None
    assert EmbeddingHead(hidden_size=32, embedding_dim=768, bias=False).linear.bias is None


def test_activation_is_configurable():
    assert isinstance(EmbeddingHead(32, 768, activation="gelu").activation, torch.nn.GELU)
    assert isinstance(EmbeddingHead(32, 768, activation=None).activation, torch.nn.Identity)
    with pytest.raises(ValueError, match="unsupported activation"):
        EmbeddingHead(32, 768, activation="swishy")


def test_warns_when_neither_norm_nor_normalisation():
    with pytest.warns(UserWarning, match="neither LayerNorm nor L2"):
        EmbeddingHead(hidden_size=32, embedding_dim=768, use_layer_norm=False, normalize=False)


def test_head_is_small_relative_to_backbone():
    """A projection head must be a tiny fraction of an 8B backbone."""
    head = EmbeddingHead(hidden_size=4096, embedding_dim=768)
    trainable, _ = count_trainable(head)
    assert trainable == 4096 * 768 + 2 * 768  # weights + LayerNorm gamma/beta
    assert trainable / 8_030_261_248 < 0.05  # < 5% of Llama 3.1 8B


# --------------------------------------------------------------------------- #
# Matryoshka truncation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dim", [256, 384, 512, 768])
def test_truncate_preserves_unit_norm(dim):
    head = EmbeddingHead(hidden_size=64, embedding_dim=1024)
    embeddings = head(torch.randn(6, 64))
    truncated = head.truncate(embeddings, dim)
    assert truncated.shape == (6, dim)
    assert torch.allclose(truncated.norm(dim=-1), torch.ones(6), atol=1e-5)


def test_truncate_matches_manual_slice_and_renormalise():
    head = EmbeddingHead(hidden_size=32, embedding_dim=1024, normalize=False)
    x = torch.randn(3, 32)
    embeddings = head(x)
    manual = l2_normalize(embeddings[:, :256])
    assert torch.allclose(head.truncate(embeddings, 256), manual, atol=1e-6)


def test_truncate_rejects_invalid_dim():
    head = EmbeddingHead(hidden_size=32, embedding_dim=768)
    with pytest.raises(ValueError, match="unsupported truncation dim"):
        head.truncate(torch.randn(2, 768), 111)


def test_truncate_rejects_dim_larger_than_vector():
    head = EmbeddingHead(hidden_size=32, embedding_dim=256)
    with pytest.raises(ValueError, match="cannot truncate"):
        head.truncate(torch.randn(2, 256), 512)


# --------------------------------------------------------------------------- #
# Gradients
# --------------------------------------------------------------------------- #


def test_head_is_differentiable():
    head = EmbeddingHead(hidden_size=32, embedding_dim=768)
    pooled = torch.randn(4, 32, requires_grad=True)
    head(pooled).sum().backward()
    assert pooled.grad is not None
    assert torch.isfinite(pooled.grad).all()
    assert head.linear.weight.grad is not None


# --------------------------------------------------------------------------- #
# Config integration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dim", SUPPORTED_EMBEDDING_DIMS)
def test_config_allows_each_supported_dim(dim):
    config = EmbeddingConfig(hidden_size=4096, embedding_dim=dim)
    assert config.embedding_dim == dim


def test_config_rejects_unsupported_dim():
    with pytest.raises(ValueError, match="embedding_dim must be one of"):
        EmbeddingConfig(embedding_dim=100)


def test_config_allows_embedding_dim_above_hidden_size():
    """A widening projection is valid and is what Matryoshka setups need."""
    config = EmbeddingConfig(hidden_size=64, embedding_dim=1024)
    assert config.embedding_dim == 1024
    head = EmbeddingHead(hidden_size=64, embedding_dim=1024)
    assert head(torch.randn(3, 64)).shape == (3, 1024)


def test_config_rejects_bad_pooling():
    with pytest.raises(ValueError, match="pooling must be one of"):
        EmbeddingConfig(pooling="cls")


def test_config_rejects_matryoshka_above_dim():
    with pytest.raises(ValueError, match="matryoshka_dims must all be"):
        EmbeddingConfig(hidden_size=4096, embedding_dim=512, matryoshka_dims=(768, 1024))


def test_config_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="unsupported dtype"):
        EmbeddingConfig(dtype="float8")


def test_config_rejects_max_length_beyond_context():
    with pytest.raises(ValueError, match="context window"):
        EmbeddingConfig(max_length=200_000)


def test_config_roundtrip_strips_token():
    config = EmbeddingConfig(token="hf_secret_value", embedding_dim=512)
    data = config.to_dict()
    assert data["token"] is None
    restored = EmbeddingConfig.from_dict(data)
    assert restored.token is None
    assert restored.embedding_dim == 512


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown embedding config keys"):
        EmbeddingConfig.from_dict({"embedding_dim": 768, "not_a_field": 1})
