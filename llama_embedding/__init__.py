"""Llama3.1-Embedding — sentence/document embeddings from Llama 3.1 Base.

An independent, open-source embedding architecture built on Meta's Llama 3.1
**base** (pretrained, non-instruct) model. This package turns decoder hidden
states into fixed-size, L2-normalised sentence vectors suitable for semantic
search and retrieval.

Quick start::

    from llama_embedding import LlamaEmbeddingModel

    model = LlamaEmbeddingModel.from_pretrained("meta-llama/Llama-3.1-8B")
    vectors = model.encode(["a search and rescue drone", "an emergency aircraft"])
    vectors.shape  # torch.Size([2, 768])

Note: ``transformers`` and PyTorch are required for real inference but *not*
for the unit tests, which run against a tiny stub backbone.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import (  # noqa: F401  (torch-free)
    DEFAULT_TEMPERATURE,
    SUPPORTED_EMBEDDING_DIMS,
    SUPPORTED_MODES,
    SUPPORTED_POOLING,
    ConfigError,
    EmbeddingConfig,
    PromptTemplates,
    TrainingConfig,
    load_config_file,
)

# The tensor/model imports below require PyTorch. Importing this package should
# not hard-fail without it (config validation and dataset tooling are useful
# standalone), so a missing torch degrades to a clear error at *use* time rather
# than an ImportError at import time.
try:
    import torch as _torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - depends on the environment
    _HAS_TORCH = False

if _HAS_TORCH:
    from .inference import SemanticSearchIndex, SearchResult, rank_documents
    from .losses import contrastive_loss, info_nce, matryoshka_info_nce, pairwise_info_nce
    from .model import (
        BackboneAdapter,
        LlamaEmbeddingModel,
        StubBackbone,
        TransformersBackbone,
    )
    from .pooling import (
        LastTokenPooling,
        MeanPooling,
        PoolingStrategy,
        WeightedMeanPooling,
        build_pooling,
        get_pooling,
    )
    from .projection import EmbeddingHead, count_trainable, l2_normalize
    from .similarity import cosine_similarity, similarity_matrix
    from .tokenizer import DummyTokenizer, build_tokenizer, tokenize_texts

    _TORCH_EXPORTS = [
        "SemanticSearchIndex",
        "SearchResult",
        "rank_documents",
        "contrastive_loss",
        "info_nce",
        "matryoshka_info_nce",
        "pairwise_info_nce",
        "BackboneAdapter",
        "LlamaEmbeddingModel",
        "StubBackbone",
        "TransformersBackbone",
        "PoolingStrategy",
        "MeanPooling",
        "LastTokenPooling",
        "WeightedMeanPooling",
        "get_pooling",
        "build_pooling",
        "EmbeddingHead",
        "l2_normalize",
        "count_trainable",
        "cosine_similarity",
        "similarity_matrix",
        "DummyTokenizer",
        "build_tokenizer",
        "tokenize_texts",
    ]
else:  # pragma: no cover - depends on the environment
    _TORCH_EXPORTS = []

__all__ = [
    "__version__",
    # config
    "EmbeddingConfig",
    "TrainingConfig",
    "PromptTemplates",
    "ConfigError",
    "load_config_file",
    "SUPPORTED_EMBEDDING_DIMS",
    "SUPPORTED_POOLING",
    "SUPPORTED_MODES",
    "DEFAULT_TEMPERATURE",
] + _TORCH_EXPORTS


def _torch_available() -> bool:
    """Whether the tensor-dependent half of this package can be used."""
    return _HAS_TORCH
