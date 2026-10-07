"""The Llama3.1-Embedding encoder.

    TEXT
      -> Tokenizer
      -> Llama 3.1 Base transformer
      -> last hidden states        [B, T, hidden_size]
      -> mask-aware pooling        [B, hidden_size]
      -> projection head           [B, embedding_dim]
      -> L2 normalisation          [B, embedding_dim]

Only hidden representations are used. The language-model head (vocabulary
logits) is explicitly bypassed: ``output_hidden_states`` / the base model's
``last_hidden_state`` is what the embedding is built from, never ``logits``.

The backbone is loaded through :class:`BackboneAdapter`, which by default wraps
``transformers.AutoModel`` (the base, non-instruct model). A tiny stub backbone
is also provided so the whole pipeline can be exercised on CPU with synthetic
tensors and without downloading 16 GB of weights.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from .config import EmbeddingConfig
from .pooling import PoolingStrategy, build_pooling
from .projection import EmbeddingHead, count_trainable
from .tokenizer import TokenizedBatch, tokenize_texts

__all__ = [
    "LlamaEmbeddingModel",
    "BackboneAdapter",
    "TransformersBackbone",
    "StubBackbone",
    "OUT_DIR_NAME",
]

OUT_DIR_NAME = "llama_embedding_out"


# --------------------------------------------------------------------------- #
# Backbones
# --------------------------------------------------------------------------- #


class BackboneAdapter(nn.Module):
    """Minimal interface the embedding model needs from a backbone."""

    hidden_size: int

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - abstract
        """Return ``[B, T, hidden_size]`` token hidden states."""
        raise NotImplementedError


class TransformersBackbone(BackboneAdapter):
    """Wraps a Hugging Face *base* Llama 3.1 model.

    ``AutoModel`` is used rather than ``AutoModelForCausalLM`` so that the
    vocabulary projection (128k logits per position) is never materialised. For
    an 8B model at batch 8 x 512 tokens that single choice saves multiple GB of
    activations per forward pass.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        hidden_size: Optional[int] = None,
        dtype: str = "float16",
        device: Optional[str] = None,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
        attn_implementation: Optional[str] = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "transformers is required to load the Llama backbone: pip install 'transformers>=4.43'"
            ) from exc

        torch_dtype = _resolve_dtype(dtype)
        kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code, "use_cache": False}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if token:
            kwargs["token"] = token
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        self._hidden_size = hidden_size or int(self.model.config.hidden_size)
        if device:
            self.model.to(device)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        # last_hidden_state: [B, T, H]. Explicitly NOT the LM logits.
        return getattr(outputs, "last_hidden_state", outputs[0])


class StubBackbone(BackboneAdapter):
    """Tiny deterministic backbone for CPU tests and pipeline smoke runs.

    Produces ``[B, T, hidden_size]`` states from token ids via an embedding
    lookup plus a fixed (seeded) linear map. It is not a language model and is
    never used for real inference — its only purpose is to let the full
    tokenize -> pool -> project -> normalise path be tested in milliseconds.
    """

    def __init__(self, vocab_size: int = 128256, hidden_size: int = 64, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.randn(vocab_size, hidden_size, generator=generator) * 0.02
            )
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(hidden_size, hidden_size, generator=generator) / 8.0)
        self._hidden_size = hidden_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(self.embedding(input_ids))
        # Mimic a causal context effect cheaply so that pooling strategies are
        # distinguishable: attenuate masked positions strongly.
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return hidden * (0.25 * mask + 0.75)


def _resolve_dtype(dtype: str) -> Optional[torch.dtype]:
    if dtype in ("auto", None, ""):
        return None
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype {dtype!r}")


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #


@dataclass
class EncodeOutput:
    """Result of :meth:`LlamaEmbeddingModel.encode`."""

    embeddings: torch.Tensor  # [N, D]
    pooled: torch.Tensor  # [N, hidden_size] (pre-projection)
    token_embeddings: Optional[torch.Tensor] = None  # [N, T, hidden_size] if requested


class LlamaEmbeddingModel(nn.Module):
    """Sentence/document encoder built on a Llama 3.1 **base** backbone.

    Example
    -------
    >>> model = LlamaEmbeddingModel(backbone=StubBackbone(hidden_size=64))  # doctest: +SKIP
    >>> vectors = model.encode(["search and rescue drone", "emergency aircraft"])
    >>> vectors.shape
    torch.Size([2, 768])
    """

    def __init__(
        self,
        backbone: BackboneAdapter,
        config: EmbeddingConfig,
        tokenizer: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.pooling: PoolingStrategy = build_pooling(config)
        self.head = EmbeddingHead(
            hidden_size=backbone.hidden_size,
            embedding_dim=config.embedding_dim,
            use_layer_norm=config.use_layer_norm,
            normalize=config.normalize,
            bias=config.bias,
            layer_norm_eps=config.layer_norm_eps,
            matryoshka_dims=config.matryoshka_dims,
        )

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        backbone_name_or_path: Optional[str] = None,
        *,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        token: Optional[str] = None,
        gradient_checkpointing: bool = False,
        backbone: Optional[BackboneAdapter] = None,
        tokenizer: Optional[Any] = None,
        head: Optional[Dict[str, torch.Tensor]] = None,
        **overrides: Any,
    ) -> "LlamaEmbeddingModel":
        """Load a backbone plus (optionally) a trained head from ``output_dir``.

        Two supported flows:

        * ``LlamaEmbeddingModel.from_pretrained("meta-llama/Llama-3.1-8B")``
          — fresh, randomly initialised projection head.
        * ``LlamaEmbeddingModel.from_pretrained(output_dir="runs/exp")``
          — loads ``embedding_head.pt`` and ``embedding_config.json`` saved by
          the trainer, then resolves the backbone id recorded there.

        Meta's Llama weights are **never** bundled with this project; they are
        fetched from Hugging Face by transformers and require the user to have
        accepted the gated licence.
        """
        if backbone is None and backbone_name_or_path is None and output_dir is None:
            raise ValueError("provide backbone_name_or_path, output_dir or backbone")

        config: Optional[EmbeddingConfig] = None
        head_state: Optional[Dict[str, torch.Tensor]] = None

        if output_dir is not None:
            config_path = os.path.join(output_dir, "embedding_config.json")
            head_path = os.path.join(output_dir, "embedding_head.pt")
            if not os.path.isfile(config_path):
                raise FileNotFoundError(
                    f"no embedding_config.json in {output_dir}. Train a checkpoint "
                    "first (python -m training.train) or pass backbone_name_or_path."
                )
            with open(config_path, "r", encoding="utf-8") as handle:
                config = EmbeddingConfig.from_dict(json.load(handle))
            if os.path.isfile(head_path):
                head_state = torch.load(head_path, map_location="cpu", weights_only=True)
            elif head is None:
                raise FileNotFoundError(
                    f"no embedding_head.pt in {output_dir}. Pass head=<state dict> to "
                    "supply one explicitly."
                )

        if config is None:
            backbone_name_or_path = backbone_name_or_path or "meta-llama/Llama-3.1-8B"
            config = EmbeddingConfig(
                backbone_name_or_path=backbone_name_or_path, **overrides
            )
        else:
            # Explicit overrides win over the stored config.
            for key, value in overrides.items():
                if value is not None and hasattr(config, key):
                    setattr(config, key, value)
            if device is not None:
                config.device = device
            if dtype is not None:
                config.dtype = dtype
            if token is not None:
                config.token = token
            config.validate()

        resolved_name = backbone_name_or_path or config.backbone_name_or_path
        if backbone is None:
            backbone = TransformersBackbone(
                resolved_name,
                # hidden_size=None so the backbone's own config wins; the stored
                # value may be stale if a different Llama size is substituted.
                dtype=config.dtype,
                device=config.device,
                token=config.token or token,
                trust_remote_code=config.trust_remote_code,
                attn_implementation=config.attn_implementation,
                gradient_checkpointing=gradient_checkpointing,
            )
            config.backbone_name_or_path = resolved_name
            config.hidden_size = backbone.hidden_size
            config.validate()

        if tokenizer is None and backbone_name_or_path is not None:
            from .tokenizer import build_tokenizer

            tokenizer = build_tokenizer(
                resolved_name,
                token=config.token or token,
                trust_remote_code=config.trust_remote_code,
            )

        model = cls(backbone=backbone, config=config, tokenizer=tokenizer)
        if head is not None:
            head_state = head
        if head_state is not None:
            model.load_head(head_state)
        return model

    # -- head persistence --------------------------------------------------- #
    def head_state_dict(self) -> Dict[str, torch.Tensor]:
        """State dict of the projection head only (safe to commit: ~a few MB)."""
        return {f"head.{k}": v.detach().cpu() for k, v in self.head.state_dict().items()}

    def load_head(self, state: Dict[str, torch.Tensor], strict: bool = True) -> None:
        cleaned = {
            k[len("head."):] if k.startswith("head.") else k: v
            for k, v in state.items()
        }
        self.head.load_state_dict(cleaned, strict=strict)

    def save_pretrained(self, output_dir: str, *, extra_metadata: Optional[Dict[str, Any]] = None) -> str:
        """Persist the head weights, the vector-architecture config and metadata.

        The Llama backbone itself is **not** saved: Meta's weights are not
        redistributed by this project. Save space is spent only on the (small)
        projection head plus LoRA adapters when present.
        """
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.head_state_dict(), os.path.join(output_dir, "embedding_head.pt"))
        self.config.save(os.path.join(output_dir, "embedding_config.json"))

        trainable, total = count_trainable(self)
        metadata: Dict[str, Any] = {
            "embedding_dim": self.config.embedding_dim,
            "hidden_size": self.config.hidden_size,
            "pooling": self.config.pooling,
            "normalize": self.config.normalize,
            "backbone": self.config.backbone_name_or_path,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_percent": (100.0 * trainable / total) if total else 0.0,
            "library": "llama3.1-embedding",
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return output_dir

    # -- freeze policies ----------------------------------------------------- #
    def set_trainable_mode(self, mode: str) -> Dict[str, Any]:
        """Apply a parameter-efficient training policy.

        ``projection_only``
            Freeze the entire backbone; train only the projection head.
        ``lora``
            Freeze the backbone base weights and train the projection head plus
            LoRA adapters on the configured target modules.
        ``full``
            Unfreeze everything (architecturally supported, very expensive for
            8B — documented, never the default).
        """
        if mode not in ("projection_only", "lora", "full"):
            raise ValueError(f"unknown training mode {mode!r}")

        for param in self.backbone.parameters():
            param.requires_grad = False

        if mode == "full":
            for param in self.backbone.parameters():
                param.requires_grad = True
            for param in self.head.parameters():
                param.requires_grad = True
        elif mode == "projection_only":
            for param in self.head.parameters():
                param.requires_grad = True
        else:  # lora
            for param in self.head.parameters():
                param.requires_grad = True
            # LoRA params were injected by the trainer via PEFT and are already
            # trainable; nothing else in the backbone should be unfrozen.

        trainable, total = count_trainable(self)
        return {
            "mode": mode,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_percent": (100.0 * trainable / total) if total else 0.0,
        }

    # -- core forward -------------------------------------------------------- #
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        return_token_embeddings: bool = False,
        return_pooled: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Encode a tokenised batch.

        Returns a dict with ``embeddings`` (``[B, D]``, L2-normalised when
        configured) and optionally ``token_embeddings`` / ``pooled``.
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, T]; got {tuple(input_ids.shape)}")
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} != input_ids {tuple(input_ids.shape)}"
            )

        hidden_states = self.backbone(input_ids, attention_mask)  # [B, T, H]
        pooled = self.pooling(hidden_states, attention_mask)  # [B, H]
        embeddings = self.head(pooled)  # [B, D]

        out: Dict[str, torch.Tensor] = {"embeddings": embeddings}
        if return_pooled:
            out["pooled"] = pooled
        if return_token_embeddings:
            out["token_embeddings"] = hidden_states
        return out

    # -- tokenised helpers --------------------------------------------------- #
    def encode_tokenized(
        self,
        batch: Union[TokenizedBatch, Dict[str, Any]],
        *,
        return_token_embeddings: bool = False,
        return_pooled: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Encode an already-tokenised batch (used by the training loop)."""
        out = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_token_embeddings=return_token_embeddings,
            return_pooled=return_pooled,
        )
        if "token_embeddings" in batch:
            out["tokens"] = batch["token_embeddings"]  # type: ignore[assignment]
        return out

    def apply_prompt(self, text: str, role: str) -> str:
        """Prepend the configured ``query``/``document`` template, if any."""
        return self.config.prompts.apply(text, role)

    # -- public encoding API -------------------------------------------------- #
    def encode(
        self,
        texts: Union[str, Sequence[str]],
        *,
        batch_size: int = 8,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None,
        device: Optional[str] = None,
        show_progress: bool = False,
        convert_to_numpy: bool = False,
        prompt_role: Optional[str] = None,
        return_pooled: bool = False,
    ) -> Union[torch.Tensor, "np.ndarray"]:  # type: ignore[name-defined]
        """Encode one or many texts into dense vectors.

        Accepts a single string or a sequence of strings and returns a
        ``[num_texts, embedding_dim]`` tensor. Handles batching internally.
        """
        if self.tokenizer is None:
            raise RuntimeError("no tokenizer attached; build the model with a tokenizer")

        single = isinstance(texts, str)
        items: List[str] = [texts] if single else list(texts)  # type: ignore[list-item]
        if not items:
            empty = torch.zeros(0, self.config.embedding_dim)
            result = empty.numpy() if convert_to_numpy else empty
            if return_pooled:
                return result, torch.zeros(0, self.config.hidden_size)
            return result
        if any(not isinstance(t, str) for t in items):
            raise TypeError("encode() expects a string or a sequence of strings")

        if prompt_role is not None:
            items = [self.apply_prompt(t, prompt_role) for t in items]

        length = max_length or self.config.max_length
        device_t = torch.device(device) if device else self._infer_device()
        self.eval()
        original_norm = self.head.normalize
        if normalize is not None:
            self.head.normalize = bool(normalize)
        try:
            chunks: List[torch.Tensor] = []
            pooled_chunks: List[torch.Tensor] = []
            with torch.inference_mode():
                for start in range(0, len(items), batch_size):
                    batch_items = items[start:start + batch_size]
                    encoded = tokenize_texts(
                        self.tokenizer, batch_items, max_length=length, return_tensors="pt"
                    )
                    input_ids = encoded["input_ids"].to(device_t)
                    attention_mask = encoded["attention_mask"].to(device_t)
                    out = self.forward(
                        input_ids,
                        attention_mask,
                        return_pooled=return_pooled,
                    )
                    chunks.append(out["embeddings"].detach().float().cpu())
                    if return_pooled:
                        pooled_chunks.append(out["pooled"].detach().float().cpu())
            embeddings = torch.cat(chunks, dim=0)
            pooled = torch.cat(pooled_chunks, dim=0) if return_pooled else None
        finally:
            self.head.normalize = original_norm

        if convert_to_numpy:
            result: Any = embeddings.numpy()
        else:
            result = embeddings
        if return_pooled:
            return result, pooled
        return result

    def encode_query(self, texts: Union[str, Sequence[str]], **kwargs: Any):
        """Encode with the configured ``query`` template applied."""
        return self.encode(texts, prompt_role="query", **kwargs)

    def encode_document(self, texts: Union[str, Sequence[str]], **kwargs: Any):
        """Encode with the configured ``document`` template applied."""
        return self.encode(texts, prompt_role="document", **kwargs)

    def _infer_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:  # pragma: no cover - parameterless stub
            return torch.device("cpu")

    # -- convenience --------------------------------------------------------- #
    @torch.no_grad()
    def similarity(self, a: Sequence[str], b: Sequence[str], **kwargs: Any) -> torch.Tensor:
        """Cosine similarity matrix between the encodings of ``a`` and ``b``."""
        from .similarity import similarity_matrix

        va = self.encode(a, **kwargs)
        vb = self.encode(b, **kwargs)
        return similarity_matrix(va, vb, metric="cosine")

    def matryoshka_prefixes(self, dim: int) -> torch.Tensor:
        """Truncate the encoder output to ``dim`` and renormalise."""
        return self.head.truncate(
            torch.eye(self.config.embedding_dim), dim
        )  # identity basis trick, used by tests
