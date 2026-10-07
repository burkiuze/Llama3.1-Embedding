"""Shared pytest fixtures and import bootstrap for the test-suite.

The unit tests must never download the 8B Llama checkpoint and must run on CPU
in seconds, so they use synthetic tensors and the :class:`StubBackbone`.

Import bootstrap: the project is a source tree (not necessarily installed), so
the repository root is prepended to ``sys.path`` here rather than in every test.

PyTorch is a hard requirement for the tensor-level tests. It is imported
*defensively* here so that a machine without torch still collects and runs the
torch-free tests (config parsing, dataset loading) instead of erroring out at
collection time. Test modules that genuinely need tensors call
``pytest.importorskip("torch")`` themselves and are reported as skipped.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_TORCH = False

TINY_HIDDEN = 64
TINY_VOCAB = 2048


# --------------------------------------------------------------------------- #
# torch-free fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def repo_root() -> str:
    return REPO_ROOT


@pytest.fixture(scope="session")
def has_torch() -> bool:
    return HAS_TORCH


# --------------------------------------------------------------------------- #
# torch fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tiny_config():
    """A valid 768-dim embedding config with a small backbone hidden size."""
    from llama_embedding.config import EmbeddingConfig

    return EmbeddingConfig(
        backbone_name_or_path="stub://tiny",
        hidden_size=TINY_HIDDEN,
        embedding_dim=768,
        max_length=32,
        dtype="float32",
    )


@pytest.fixture
def stub_tokenizer():
    from llama_embedding.tokenizer import DummyTokenizer

    return DummyTokenizer(vocab_size=TINY_VOCAB)


@pytest.fixture
def tiny_model(tiny_config, stub_tokenizer):
    from llama_embedding.model import LlamaEmbeddingModel, StubBackbone

    torch.manual_seed(0)
    return LlamaEmbeddingModel(
        backbone=StubBackbone(vocab_size=TINY_VOCAB, hidden_size=TINY_HIDDEN, seed=0),
        config=tiny_config,
        tokenizer=stub_tokenizer,
    )


@pytest.fixture
def hidden_states():
    """``[4, 6, 8]`` hidden states."""
    torch.manual_seed(1234)
    return torch.randn(4, 6, 8, dtype=torch.float32)


@pytest.fixture
def attention_mask():
    """Right-padded mask: lengths 6, 4, 2, 1."""
    return torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
