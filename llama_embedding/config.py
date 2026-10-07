"""Configuration objects for Llama3.1-Embedding.

Design goals
------------
* One place that validates *every* user-facing knob (dims, pooling, modes...).
* YAML configs, but **no hard dependency on PyYAML**: a small, well tested
  parser for the restricted YAML subset used by ``configs/`` is provided so the
  project stays importable in minimal environments.
* Serialisable back and forth so a training run can be reproduced exactly from
  the config that produced it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "SUPPORTED_EMBEDDING_DIMS",
    "SUPPORTED_POOLING",
    "SUPPORTED_MODES",
    "DEFAULT_TEMPERATURE",
    "EmbeddingConfig",
    "TrainingConfig",
    "ConfigError",
    "load_yaml",
    "load_config_file",
    "config_from_dict",
]

# --------------------------------------------------------------------------- #
# Enumerations / documented defaults
# --------------------------------------------------------------------------- #

#: Output dimensions the projection head is validated against.
SUPPORTED_EMBEDDING_DIMS: Tuple[int, ...] = (256, 384, 512, 768, 1024)

#: Pooling strategies exposed by :mod:`llama_embedding.pooling`.
SUPPORTED_POOLING: Tuple[str, ...] = ("mean", "last_token", "weighted_mean")

#: Parameter-efficient training modes.
SUPPORTED_MODES: Tuple[str, ...] = ("projection_only", "lora", "full")

#: Software default temperature for InfoNCE.
#:
#: This is **a default, not a tuned value**. It is the temperature used by
#: sentence-transformers' MultipleNegativesRankingLoss default, chosen because it
#: is a widely used, documented starting point. It is explicitly tunable
#: (``temperature`` in the training config) and should be swept on a validation
#: split before being treated as an optimal value.
DEFAULT_TEMPERATURE: float = 0.02


class ConfigError(ValueError):
    """Raised when a configuration value is invalid or contradictory."""


# --------------------------------------------------------------------------- #
# Minimal YAML subset parser (PyYAML is optional)
# --------------------------------------------------------------------------- #

_TRUE = {"true", "yes", "on"}
_FALSE = {"false", "no", "off"}
_NULL = {"null", "none", "~"}


def _strip_comment(line: str) -> str:
    """Remove a trailing ``#`` comment while respecting quoted strings."""
    out: List[str] = []
    quote: Optional[str] = None
    prev = ""
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote and prev != "\\":
                quote = None
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#" and (not out or out[-1] in " \t"):
            break
        else:
            out.append(ch)
        prev = ch
    return "".join(out).rstrip()


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    low = token.lower()
    if low in _NULL:
        return None
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(p) for p in _split_flow(inner)]
    if token.startswith("{") and token.endswith("}"):
        inner = token[1:-1].strip()
        result: Dict[str, Any] = {}
        if inner:
            for part in _split_flow(inner):
                k, _, v = part.partition(":")
                result[str(_parse_scalar(k))] = _parse_scalar(v)
        return result
    if re.fullmatch(r"[+-]?\d+", token):
        return int(token)
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?", token):
        return float(token)
    return token


def _split_flow(text: str) -> List[str]:
    """Split ``a, b, {c: d}`` on top-level commas only."""
    parts: List[str] = []
    depth = 0
    quote: Optional[str] = None
    current: List[str] = []
    for ch in text:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            current.append(ch)
        elif ch in "[{":
            depth += 1
            current.append(ch)
        elif ch in "]}":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return [p.strip() for p in parts]


def _parse_block(lines: Sequence[Tuple[int, str]], index: int, indent: int) -> Tuple[Any, int]:
    """Parse a YAML block starting at ``index`` with the given indentation."""
    # Decide whether this block is a list or a mapping.
    if index < len(lines) and lines[index][1].startswith("- "):
        result_list: List[Any] = []
        while index < len(lines):
            cur_indent, text = lines[index]
            if cur_indent < indent or not text.startswith("- "):
                break
            item_text = text[2:].strip()
            if item_text == "":
                index += 1
                if index < len(lines) and lines[index][0] > cur_indent:
                    value, index = _parse_block(lines, index, lines[index][0])
                else:
                    value = None
                result_list.append(value)
            elif ":" in item_text and not item_text.startswith(("\"", "'")):
                # inline mapping start: "- key: value"
                sub_lines = [(cur_indent + 2, item_text)]
                index += 1
                while index < len(lines) and lines[index][0] > cur_indent:
                    sub_lines.append(lines[index])
                    index += 1
                value, _ = _parse_block(sub_lines, 0, cur_indent + 2)
                result_list.append(value)
            else:
                result_list.append(_parse_scalar(item_text))
                index += 1
        return result_list, index

    result_map: Dict[str, Any] = {}
    while index < len(lines):
        cur_indent, text = lines[index]
        if cur_indent < indent:
            break
        if text.startswith("- "):
            break
        key_part, sep, value_part = text.partition(":")
        if not sep:
            raise ConfigError(f"Malformed YAML line (expected 'key: value'): {text!r}")
        key = str(_parse_scalar(key_part.strip()))
        value_text = value_part.strip()
        index += 1
        if value_text == "":
            if index < len(lines) and lines[index][0] > cur_indent:
                child, index = _parse_block(lines, index, lines[index][0])
                result_map[key] = child
            elif (
                index < len(lines)
                and lines[index][0] == cur_indent
                and lines[index][1].startswith("- ")
            ):
                child, index = _parse_block(lines, index, cur_indent)
                result_map[key] = child
            else:
                result_map[key] = None
        else:
            result_map[key] = _parse_scalar(value_text)
    return result_map, index


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file.

    Uses PyYAML when it is installed (authoritative), otherwise falls back to the
    built-in parser which supports the subset used by this repository: nested
    mappings, block lists, flow lists, comments and scalar types.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read()
    try:  # pragma: no cover - depends on the environment
        import yaml  # type: ignore

        return yaml.safe_load(raw) or {}
    except ImportError:
        pass

    lines: List[Tuple[int, str]] = []
    for raw_line in raw.splitlines():
        if raw_line.strip().startswith("#"):
            continue
        cleaned = _strip_comment(raw_line)
        if not cleaned.strip():
            continue
        if cleaned.strip() in ("---", "..."):
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        lines.append((indent, cleaned.strip()))
    if not lines:
        return {}
    parsed, _ = _parse_block(lines, 0, lines[0][0])
    return parsed if isinstance(parsed, dict) else {}


def load_config_file(path: str) -> Dict[str, Any]:
    """Load a ``.json`` or ``.yaml``/``.yml`` config file into a plain dict."""
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path}")
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if path.endswith((".yaml", ".yml")):
        return load_yaml(path)
    raise ConfigError(f"Unsupported config extension: {path} (use .yaml, .yml or .json)")


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class PromptTemplates:
    """Retrieval prefixes.

    Kept configurable (and language-agnostic) on purpose: no English-only
    instruction is hard coded anywhere in the vector path. ``query`` and
    ``document`` default to the symmetric convention used by
    sentence-transformers-style models, but a project (or a language) can
    override them in config without touching any code.
    """

    query: str = ""
    document: str = ""

    def apply(self, text: str, role: str) -> str:
        if role not in ("query", "document"):
            raise ConfigError(f"role must be 'query' or 'document', got {role!r}")
        prefix = getattr(self, role)
        return f"{prefix}{text}" if prefix else text

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "PromptTemplates":
        data = data or {}
        return cls(query=str(data.get("query", "") or ""), document=str(data.get("document", "") or ""))


@dataclass
class EmbeddingConfig:
    """Everything that defines the *vector architecture*.

    This object fully determines the encoder output: backbone, pooling,
    projection, normalisation, output dimension and prompt handling. Two
    :class:`EmbeddingConfig` objects with equal fields produce equal embeddings
    for the same weights.
    """

    backbone_name_or_path: str = "meta-llama/Llama-3.1-8B"
    pooling: str = "mean"
    embedding_dim: int = 768
    hidden_size: int = 4096  # Llama 3.1 8B; overridden from the backbone config
    max_length: int = 512
    normalize: bool = True
    use_layer_norm: bool = True
    layer_norm_eps: float = 1e-12
    bias: bool = False
    dtype: str = "float16"
    device: Optional[str] = None
    trust_remote_code: bool = False
    token: Optional[str] = None  # never persisted to disk; supplied by the user
    prompts: PromptTemplates = field(default_factory=PromptTemplates)
    matryoshka_dims: Tuple[int, ...] = ()
    attn_implementation: Optional[str] = None
    use_cache: bool = False

    def __post_init__(self) -> None:
        self.validate()

    # -- validation -------------------------------------------------------- #
    def validate(self) -> "EmbeddingConfig":
        if self.pooling not in SUPPORTED_POOLING:
            raise ConfigError(
                f"pooling must be one of {SUPPORTED_POOLING}, got {self.pooling!r}"
            )
        if self.embedding_dim not in SUPPORTED_EMBEDDING_DIMS:
            raise ConfigError(
                f"embedding_dim must be one of {SUPPORTED_EMBEDDING_DIMS}, got {self.embedding_dim!r}"
            )
        if self.hidden_size <= 0:
            raise ConfigError(f"hidden_size must be positive, got {self.hidden_size}")
        # NOTE: embedding_dim > hidden_size is intentionally allowed. A widening
        # Linear(hidden -> embedding_dim) is valid and is what Matryoshka-style
        # setups want (project a small backbone up to a large embedding space).
        if self.max_length <= 0:
            raise ConfigError(f"max_length must be positive, got {self.max_length}")
        if self.max_length > 131072:
            raise ConfigError(
                f"max_length {self.max_length} exceeds the Llama 3.1 context window (131072)"
            )
        if self.dtype not in ("float32", "float16", "bfloat16", "auto"):
            raise ConfigError(f"unsupported dtype {self.dtype!r}")
        dims = tuple(self.matryoshka_dims or ())
        bad = [d for d in dims if d not in SUPPORTED_EMBEDDING_DIMS]
        if bad:
            raise ConfigError(f"matryoshka_dims contains unsupported values: {bad}")
        if any(d > self.embedding_dim for d in dims):
            raise ConfigError(
                "matryoshka_dims must all be <= embedding_dim "
                f"(got {dims} vs {self.embedding_dim})"
            )
        self.matryoshka_dims = dims
        return self

    # -- (de)serialisation ------------------------------------------------- #
    def to_dict(self, *, include_secrets: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        data["prompts"] = asdict(self.prompts)
        data["matryoshka_dims"] = list(self.matryoshka_dims)
        if not include_secrets:
            # A HF token must never reach a checkpoint, a log or Git.
            data["token"] = None
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EmbeddingConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Unknown embedding config keys: {sorted(unknown)}")
        kwargs = dict(data)
        prompts = kwargs.pop("prompts", None)
        kwargs["prompts"] = PromptTemplates.from_dict(prompts)
        kwargs["matryoshka_dims"] = tuple(kwargs.get("matryoshka_dims") or ())
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str) -> "EmbeddingConfig":
        return cls.from_dict(load_config_file(path))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")


@dataclass
class TrainingConfig:
    """Optimisation, data and bookkeeping settings for contrastive training."""

    # data
    train_file: str = "data/sample/train.jsonl"
    eval_file: Optional[str] = None
    max_length: int = 256
    validation_split: float = 0.1
    seed: int = 42
    shuffle: bool = True
    dedupe: bool = True
    num_hard_negatives: int = 1
    hard_negative_strategy: str = "provided"  # provided | random | mixed

    # optimisation
    mode: str = "projection_only"
    batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    head_learning_rate: Optional[float] = None  # defaults to learning_rate
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_epochs: int = 1
    max_steps: int = -1
    max_grad_norm: float = 1.0
    temperature: float = DEFAULT_TEMPERATURE
    similarity_metric: str = "cosine"
    symmetric_loss: bool = False
    matryoshka_loss_weight: float = 0.0
    label_smoothing: float = 0.0

    # runtime / hardware
    mixed_precision: str = "auto"  # auto | no | fp16 | bf16
    gradient_checkpointing: bool = False
    device: Optional[str] = None
    num_workers: int = 0
    log_every: int = 1
    eval_every: int = 50
    save_every: int = 50
    fail_on_nan: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # checkpointing
    output_dir: str = "runs/llama31-embedding"
    resume_from: Optional[str] = None
    save_total_limit: int = 2

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "TrainingConfig":
        if self.mode not in SUPPORTED_MODES:
            raise ConfigError(f"mode must be one of {SUPPORTED_MODES}, got {self.mode!r}")
        if self.batch_size <= 0:
            raise ConfigError("batch_size must be >= 1")
        if self.eval_batch_size <= 0:
            raise ConfigError("eval_batch_size must be >= 1")
        if self.gradient_accumulation_steps <= 0:
            raise ConfigError("gradient_accumulation_steps must be >= 1")
        if self.learning_rate <= 0:
            raise ConfigError("learning_rate must be > 0")
        if self.head_learning_rate is not None and self.head_learning_rate <= 0:
            raise ConfigError("head_learning_rate must be > 0")
        if self.temperature <= 0:
            raise ConfigError("temperature must be > 0")
        if self.temperature < 1e-3:
            raise ConfigError("temperature is implausibly small (< 1e-3); the loss will saturate")
        if self.similarity_metric not in ("cosine", "dot", "euclidean"):
            raise ConfigError(f"unsupported similarity_metric {self.similarity_metric!r}")
        if self.mixed_precision not in ("auto", "no", "fp16", "bf16"):
            raise ConfigError(f"unsupported mixed_precision {self.mixed_precision!r}")
        if not (0.0 <= self.validation_split < 1.0):
            raise ConfigError("validation_split must be in [0, 1)")
        if self.hard_negative_strategy not in ("provided", "random", "mixed"):
            raise ConfigError(
                f"unsupported hard_negative_strategy {self.hard_negative_strategy!r}"
            )
        if self.num_hard_negatives < 0:
            raise ConfigError("num_hard_negatives must be >= 0")
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ConfigError("label_smoothing must be in [0, 1)")
        if self.max_epochs <= 0 and self.max_steps <= 0:
            raise ConfigError("either max_epochs or max_steps must be positive")
        self.lora_target_modules = tuple(self.lora_target_modules)
        return self

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["lora_target_modules"] = list(self.lora_target_modules)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Unknown training config keys: {sorted(unknown)}")
        kwargs = dict(data)
        kwargs["lora_target_modules"] = tuple(kwargs.get("lora_target_modules") or ())
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str) -> "TrainingConfig":
        return cls.from_dict(load_config_file(path))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")


def config_from_dict(data: Mapping[str, Any]) -> Tuple[EmbeddingConfig, TrainingConfig]:
    """Split a combined ``{embedding: ..., training: ...}`` mapping."""
    emb_data = data.get("embedding", {}) or {}
    train_data = data.get("training", {}) or {}
    return EmbeddingConfig.from_dict(emb_data), TrainingConfig.from_dict(train_data)
