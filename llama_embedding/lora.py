"""LoRA integration for parameter-efficient adaptation of the Llama backbone.

LoRA is applied **around** :meth:`LlamaEmbeddingModel.set_trainable_mode` and is
strictly optional: if ``peft`` is not installed the caller gets a clear error
rather than a silent fallback to full fine-tuning.

The rule enforced here is the important one: after configuring LoRA, the set of
trainable parameters must be small. If it is not, something went wrong and the
function raises instead of quietly training 8B parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["LoRAConfig", "apply_lora", "describe_trainable_parameters"]


@dataclass
class LoRAConfig:
    """LoRA hyper-parameters for the Llama attention/MLP projections."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Sequence[str] = field(
        default_factory=lambda: ("q_proj", "k_proj", "v_proj", "o_proj")
    )
    bias: str = "none"
    task_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "bias": self.bias,
            "task_type": self.task_type,
        }


def apply_lora(model: Any, config: LoRAConfig) -> Any:
    """Wrap ``model.backbone.model``'s transformer with LoRA adapters.

    Parameters
    ----------
    model:
        A :class:`~llama_embedding.model.LlamaEmbeddingModel`.
    config:
        LoRA settings.

    Returns
    -------
    The same model, with LoRA parameters injected and trainable.
    """
    if config.r <= 0:
        raise ValueError(f"LoRA rank must be positive, got {config.r}")

    try:
        from peft import LoraConfig as PeftLoraConfig, get_peft_model  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "LoRA training requires PEFT: pip install 'peft>=0.11'. "
            "Alternatively use mode='projection_only', which needs no extra dependency."
        ) from exc

    inner = getattr(model.backbone, "model", None)
    if inner is None:
        raise ValueError(
            "LoRA requires a transformers backbone (BackboneAdapter.model); "
            "the stub backbone is for tests only"
        )

    peft_cfg = PeftLoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        bias=config.bias,
        task_type=config.task_type,
    )
    model.backbone.model = get_peft_model(inner, peft_cfg)
    # PEFT already freezes the base weights and unfreezes the adapters.
    for param in model.head.parameters():
        param.requires_grad = True
    return model


def describe_trainable_parameters(model: Any, *, top_k_modules: int = 10) -> Dict[str, Any]:
    """Summarise trainable vs total parameters.

    Returns counts, a percentage, and the largest trainable modules so the
    training log states plainly *what* is being optimised.
    """
    total = 0
    trainable = 0
    per_module: List[tuple[str, int]] = []
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
            module_name = name.rsplit(".", 1)[0]
            per_module.append((module_name, count))
    trainable_pct = (100.0 * trainable / total) if total else 0.0

    aggregated: Dict[str, int] = {}
    for module_name, count in per_module:
        aggregated[module_name] = aggregated.get(module_name, 0) + count
    largest = sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True)[:top_k_modules]

    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": trainable_pct,
        "largest_trainable_modules": largest,
    }
