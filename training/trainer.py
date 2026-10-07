"""Contrastive trainer for Llama3.1-Embedding.

Implements a self-contained training loop (no ``Trainer`` subclass) so that
every stability control the project promises is explicit and inspectable:

* parameter-efficient modes (``projection_only`` / ``lora`` / ``full``)
* gradient accumulation, gradient clipping, mixed precision
* gradient checkpointing, LoRA, dtype selection
* NaN/Inf loss detection that **aborts** instead of continuing
* checkpointing, resume, best-checkpoint tracking
* seeded, reproducible iteration and logging

A full loop over an 8B backbone needs GPU hardware; this module is written so
that the same code path runs against a tiny stub backbone on CPU, which is what
the test-suite and smoke script exercise.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from llama_embedding.config import EmbeddingConfig, TrainingConfig
from llama_embedding.losses import contrastive_loss
from llama_embedding.lora import LoRAConfig, apply_lora, describe_trainable_parameters
from llama_embedding.model import LlamaEmbeddingModel

from .collator import ContrastiveCollator
from .dataset import ContrastiveDataset

__all__ = ["Trainer", "TrainerState", "set_seed", "NonFiniteLossError"]


class NonFiniteLossError(RuntimeError):
    """Raised when the loss or gradients become NaN/Inf during training."""


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs for reproducible runs."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA here
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainerState:
    """Mutable training progress, serialised into every checkpoint."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float = math.inf
    best_step: int = 0
    epochs_without_improvement: int = 0
    log_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainerState":
        known = {k: v for k, v in (data or {}).items() if k in cls.__annotations__}
        return cls(**known)


class Trainer:
    """Contrastive trainer.

    Parameters
    ----------
    model:
        The :class:`LlamaEmbeddingModel` to train.
    train_dataset:
        Contrastive training data.
    training_config:
        Optimisation + bookkeeping settings.
    eval_dataset:
        Optional validation data (drives best-checkpoint tracking).
    tokenizer:
        Tokenizer used by the collator.
    embedding_config:
        Vector architecture config (for prompts / dims).
    """

    def __init__(
        self,
        model: LlamaEmbeddingModel,
        train_dataset: ContrastiveDataset,
        training_config: TrainingConfig,
        *,
        eval_dataset: Optional[ContrastiveDataset] = None,
        tokenizer: Any = None,
        embedding_config: Optional[EmbeddingConfig] = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_config = training_config
        self.tokenizer = tokenizer if tokenizer is not None else model.tokenizer
        self.embedding_config = embedding_config or model.config
        self.state = TrainerState()

        if self.tokenizer is None:
            raise ValueError("a tokenizer is required to build batches")

        set_seed(self.training_config.seed)
        self.device = self._resolve_device()
        self.state_dict_saver: Dict[str, Any] = {}

        self._configure_trainable_parameters()
        self._log_parameter_summary()

        self.collator = ContrastiveCollator(
            tokenizer=self.tokenizer,
            max_length=self.training_config.max_length,
            prompts=self.embedding_config.prompts,
        )
        self.train_loader = self._build_loader(self.train_dataset, self.training_config.batch_size, shuffle=True)

    # -- setup ---------------------------------------------------------------- #
    def _resolve_device(self) -> torch.device:
        if self.training_config.device:
            return torch.device(self.training_config.device)
        if torch.cuda.is_available():  # pragma: no cover
            return torch.device("cuda")
        return torch.device("cpu")

    def _configure_trainable_parameters(self) -> None:
        mode = self.training_config.mode
        if mode == "lora":
            lora_cfg = LoRAConfig(
                r=self.training_config.lora_r,
                alpha=self.training_config.lora_alpha,
                dropout=self.training_config.lora_dropout,
                target_modules=self.training_config.lora_target_modules,
            )
            apply_lora(self.model, lora_cfg)
        info = self.model.set_trainable_mode(mode)
        self.parameter_summary = describe_trainable_parameters(self.model)

        # Guard: never silently train the entire 8B backbone. Only the explicit
        # "full" mode may exceed a small fraction of trainable parameters.
        if mode != "full":
            total = info["total_parameters"]
            pct = info["trainable_percent"]
            if total > 0 and pct > 50.0:
                raise RuntimeError(
                    f"mode={mode!r} left {pct:.2f}% of parameters trainable, which is "
                    "inconsistent with parameter-efficient training. Refusing to proceed."
                )
        self.mode_info = info

        if self.training_config.gradient_checkpointing and hasattr(self.model.backbone, "model"):
            try:
                self.model.backbone.model.gradient_checkpointing_enable()
            except Exception:  # pragma: no cover - depends on the backbone
                pass

    def _log_parameter_summary(self) -> None:
        info = self.parameter_summary
        print("=" * 68)
        print("Parameter efficiency")
        print("=" * 68)
        print(f"  mode                 : {self.training_config.mode}")
        print(f"  trainable parameters : {info['trainable_parameters']:,}")
        print(f"  total parameters     : {info['total_parameters']:,}")
        print(f"  percent trainable    : {info['trainable_percent']:.4f}%")
        top = info.get("largest_trainable_modules") or []
        if top:
            print("  largest trainable modules:")
            for name, count in top:
                print(f"    - {name}: {count:,}")
        print("=" * 68)

    def _build_loader(self, dataset: ContrastiveDataset, batch_size: int, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,  # ContrastiveDataset implements __len__/__getitem__
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collator,
            num_workers=self.training_config.num_workers,
            drop_last=shuffle and len(dataset) > batch_size,
        )

    # -- precision / optim ---------------------------------------------------- #
    def _autocast(self):
        mp = self.training_config.mixed_precision
        if mp == "no" or self.device.type != "cuda":  # pragma: no branch
            return torch.autocast(device_type=self.device.type, enabled=False)
        dtype = torch.float16 if mp == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        head_params = [p for p in self.model.head.parameters() if p.requires_grad]
        other_params = [
            p
            for name, p in self.model.named_parameters()
            if p.requires_grad and not name.startswith("head.")
        ]
        head_lr = self.training_config.head_learning_rate or self.training_config.learning_rate
        groups: List[Dict[str, Any]] = []
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
        if other_params:
            groups.append({"params": other_params, "lr": self.training_config.learning_rate})
        if not groups:
            raise RuntimeError("no trainable parameters; check the training mode")
        return torch.optim.AdamW(
            groups,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

    def _build_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: int):
        from torch.optim.lr_scheduler import LambdaLR

        warmup_steps = int(num_training_steps * self.training_config.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            return max(0.0, (num_training_steps - step) / max(1, num_training_steps - warmup_steps))

        return LambdaLR(optimizer, lr_lambda)

    # -- forward / loss -------------------------------------------------------- #
    def _encode(self, encoded: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model(
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device),
        )
        return out["embeddings"]

    def _encode_negatives(self, negative: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode ``[B, K, T]`` negatives -> ``[B, K, D]``."""
        input_ids = negative["input_ids"].to(self.device)  # [B, K, T]
        attention_mask = negative["attention_mask"].to(self.device)
        b, k, t = input_ids.shape
        flat_ids = input_ids.reshape(b * k, t)
        flat_mask = attention_mask.reshape(b * k, t)
        out = self.model(flat_ids, flat_mask)
        return out["embeddings"].reshape(b, k, -1)

    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        query_emb = self._encode(batch["query"])
        positive_emb = self._encode(batch["positive"])

        negatives: Optional[torch.Tensor] = None
        if batch.get("num_negatives", 0) > 0 and "negative" in batch:
            negatives = self._encode_negatives(batch["negative"])

        matryoshka_dims = tuple(self.embedding_config.matryoshka_dims or ())
        losses = contrastive_loss(
            query_emb,
            positive_emb,
            negatives=negatives,
            matryoshka_dims=matryoshka_dims,
            temperature=self.training_config.temperature,
            metric=self.training_config.similarity_metric,
            normalize=True,
            symmetric=self.training_config.symmetric_loss,
            label_smoothing=self.training_config.label_smoothing,
            hard_negative_weight=0.5 if negatives is not None else 0.0,
        )
        return losses

    # -- step ------------------------------------------------------------------ #
    def _micro_step(self, batch: Dict[str, Any], scale: float) -> Dict[str, float]:
        """Forward + backward on one micro-batch, loss pre-scaled for accumulation.

        Gradient clipping and the optimiser step happen once per *optimiser* step
        (see :meth:`_optimizer_step`), not per micro-batch, so accumulation
        actually reproduces a larger effective batch.
        """
        losses = self.compute_loss(batch)
        loss = losses["loss"]
        if not torch.isfinite(loss):
            raise NonFiniteLossError(
                f"loss became {loss.item()} at step {self.state.global_step}; "
                "aborting to avoid corrupting the checkpoint"
            )
        # Scaling by 1/accum_steps makes the accumulated gradient equal the mean
        # over the full effective batch.
        (loss * scale).backward()
        return {k: float(v.detach().cpu()) for k, v in losses.items()}

    def _optimizer_step(self, optimizer, scheduler) -> float:
        """Clip gradients, step the optimiser and advance the schedule."""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=self.training_config.max_grad_norm,
        )
        if not torch.isfinite(grad_norm):
            raise NonFiniteLossError(
                f"gradient norm became {float(grad_norm)} at step "
                f"{self.state.global_step}; aborting"
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return float(grad_norm)
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Mean validation loss (and accuracy proxy) over the eval dataset."""
        if self.eval_dataset is None or len(self.eval_dataset) == 0:
            return {}
        self.model.eval()
        loader = self._build_loader(self.eval_dataset, self.training_config.eval_batch_size, shuffle=False)
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for batch in loader:
            # One forward pass per side; reused for both loss and accuracy.
            query_emb = self._encode(batch["query"])
            positive_emb = self._encode(batch["positive"])
            negatives: Optional[torch.Tensor] = None
            if batch.get("num_negatives", 0) > 0 and "negative" in batch:
                negatives = self._encode_negatives(batch["negative"])

            losses = contrastive_loss(
                query_emb,
                positive_emb,
                negatives=negatives,
                matryoshka_dims=tuple(self.embedding_config.matryoshka_dims or ()),
                temperature=self.training_config.temperature,
                metric=self.training_config.similarity_metric,
                normalize=True,
                symmetric=self.training_config.symmetric_loss,
                label_smoothing=self.training_config.label_smoothing,
                hard_negative_weight=0.5 if negatives is not None else 0.0,
            )
            total_loss += float(losses["loss"].detach().cpu()) * batch["size"]

            # In-batch top-1 accuracy is a cheap proxy for retrieval quality.
            sim = (query_emb @ positive_emb.T) / self.training_config.temperature
            preds = sim.argmax(dim=-1)
            target = torch.arange(query_emb.shape[0], device=query_emb.device)
            total_correct += int((preds == target).sum().cpu())
            total_examples += batch["size"]
        self.model.train()
        if total_examples == 0:
            return {}
        return {
            "eval_loss": total_loss / total_examples,
            "eval_inbatch_accuracy": total_correct / total_examples,
        }

    # -- checkpointing ------------------------------------------------------------ #
    def _ensure_output_dir(self) -> str:
        os.makedirs(self.training_config.output_dir, exist_ok=True)
        return self.training_config.output_dir

    def save_checkpoint(self, *, is_best: bool = False) -> str:
        """Save head weights (and LoRA adapters) plus trainer state."""
        output_dir = self._ensure_output_dir()
        tag = "best" if is_best else f"step-{self.state.global_step}"
        path = os.path.join(output_dir, f"checkpoint-{tag}")
        os.makedirs(path, exist_ok=True)

        payload = {
            "head": self.model.head_state_dict(),
            "state": self.state.to_dict(),
            "training_config": self.training_config.to_dict(),
        }
        torch.save(payload, os.path.join(path, "checkpoint.pt"))

        if is_best:
            self.model.save_pretrained(
                output_dir,
                extra_metadata={
                    "step": self.state.global_step,
                    "epoch": self.state.epoch,
                    "mode": self.training_config.mode,
                    "best_metric": self.state.best_metric,
                    "best_step": self.state.best_step,
                },
            )
        return path

    def load_checkpoint(self, path: str) -> None:
        """Resume model head + trainer state from a checkpoint directory or file."""
        if os.path.isdir(path):
            path = os.path.join(path, "checkpoint.pt")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_head(payload["head"])
        self.state = TrainerState.from_dict(payload.get("state", {}))
        print(f"Resumed from checkpoint at global_step={self.state.global_step}")

    # -- main loop ------------------------------------------------------------------ #
    def train(self) -> Dict[str, Any]:
        """Run the full training loop and return a summary dict."""
        cfg = self.training_config
        accum = max(1, cfg.gradient_accumulation_steps)
        micro_steps_per_epoch = max(1, len(self.train_loader))
        micro_total = micro_steps_per_epoch * cfg.max_epochs
        if cfg.max_steps > 0:
            total_steps = cfg.max_steps          # counted in *optimiser* steps
        else:
            total_steps = max(1, micro_total // accum)

        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer, total_steps)
        self.model.to(self.device)
        self.model.train()

        start_time = time.time()
        should_stop = False
        micro_index = 0            # micro-batch counter across the whole run
        optimizer.zero_grad(set_to_none=True)
        pending: Dict[str, float] = {}

        for epoch in range(cfg.max_epochs):
            if should_stop:
                break
            self.state.epoch = epoch
            for batch in self.train_loader:
                stats = self._micro_step(batch, scale=1.0 / accum)
                for key, value in stats.items():
                    pending[key] = pending.get(key, 0.0) + value / accum
                micro_index += 1

                is_last_micro = micro_index % accum == 0
                # Also flush on the final (possibly partial) accumulation window
                # so no computed gradient is silently discarded at epoch end.
                if not (is_last_micro or micro_index >= micro_total):
                    continue

                grad_norm = self._optimizer_step(optimizer, scheduler)
                optimizer.zero_grad(set_to_none=True)
                self.state.global_step += 1

                stats = {**pending, "grad_norm": grad_norm}
                pending = {}

                if cfg.log_every and self.state.global_step % cfg.log_every == 0:
                    entry = {
                        "step": self.state.global_step,
                        "epoch": epoch,
                        "loss": stats["loss"],
                        "grad_norm": stats["grad_norm"],
                        "accum_steps": accum,
                        "elapsed": time.time() - start_time,
                    }
                    self.state.log_history.append(entry)
                    extra = " ".join(
                        f"{k}={v:.4f}" for k, v in stats.items() if k not in ("loss", "grad_norm")
                    )
                    print(
                        f"step {self.state.global_step:5d}/{total_steps} "
                        f"loss={stats['loss']:.4f} grad_norm={stats['grad_norm']:.3f} {extra}"
                    )

                if cfg.eval_every and self.state.global_step % cfg.eval_every == 0:
                    metrics = self.evaluate()
                    if metrics:
                        print(f"  [eval] step {self.state.global_step}: {metrics}")
                        self._track_best(metrics["eval_loss"])
                        self.state.log_history.append({"step": self.state.global_step, **metrics})

                if cfg.save_every and self.state.global_step % cfg.save_every == 0:
                    self.save_checkpoint()

                if cfg.max_steps > 0 and self.state.global_step >= cfg.max_steps:
                    should_stop = True
                    break
            if should_stop:
                break

        # Final validation + best checkpoint materialisation.
        final_metrics = self.evaluate()
        if final_metrics:
            print(f"[final eval] {final_metrics}")
            self._track_best(final_metrics["eval_loss"])
            self.state.log_history.append({"step": self.state.global_step, **final_metrics})

        # Always write the best/last head out so inference has something to load.
        self.save_checkpoint(is_best=True)
        history_path = os.path.join(self._ensure_output_dir(), "train_log.jsonl")
        with open(history_path, "w", encoding="utf-8") as handle:
            for entry in self.state.log_history:
                handle.write(json.dumps(entry) + "\n")

        return {
            "global_step": self.state.global_step,
            "best_metric": self.state.best_metric,
            "best_step": self.state.best_step,
            "final_metrics": final_metrics,
            "output_dir": self.training_config.output_dir,
            "parameter_summary": self.parameter_summary,
            "wall_time_seconds": time.time() - start_time,
        }

    def _track_best(self, metric: float) -> None:
        if metric < self.state.best_metric:
            self.state.best_metric = metric
            self.state.best_step = self.state.global_step
            self.state.epochs_without_improvement = 0
            self.save_checkpoint(is_best=True)
        else:
            self.state.epochs_without_improvement += 1
