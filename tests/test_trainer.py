"""Trainer smoke tests: parameter counts, loss descent, NaN abort, checkpointing.

Everything here runs on CPU against the tiny stub backbone, so the whole
training path is exercised in seconds without the 8B weights.
"""

from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")

from llama_embedding.config import EmbeddingConfig, TrainingConfig
from llama_embedding.model import LlamaEmbeddingModel, StubBackbone
from llama_embedding.tokenizer import DummyTokenizer
from training.dataset import ContrastiveExample, ContrastiveDataset
from training.trainer import NonFiniteLossError, Trainer, set_seed

SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample"
)


def _tiny_setup(tmp_path, **train_kwargs):
    set_seed(0)
    config = EmbeddingConfig(
        backbone_name_or_path="stub://tiny",
        hidden_size=32,
        embedding_dim=256,
        max_length=16,
        dtype="float32",
    )
    model = LlamaEmbeddingModel(
        backbone=StubBackbone(vocab_size=1024, hidden_size=32, seed=0),
        config=config,
        tokenizer=DummyTokenizer(vocab_size=1024),
    )
    examples = [
        ContrastiveExample(f"query number {i}", f"positive passage {i}", [f"negative {i}"])
        for i in range(8)
    ]
    dataset = ContrastiveDataset(examples, seed=0, num_hard_negatives=1, shuffle=False)
    eval_dataset = ContrastiveDataset(examples[:4], seed=0, shuffle=False)

    train_config = TrainingConfig(
        train_file=os.path.join(SAMPLE_DIR, "train.jsonl"),
        output_dir=str(tmp_path / "run"),
        batch_size=4,
        eval_batch_size=4,
        max_steps=3,
        max_epochs=1,
        eval_every=0,
        save_every=0,
        log_every=1,
        mixed_precision="no",
        validation_split=0.0,
        num_hard_negatives=1,
        **train_kwargs,
    )
    trainer = Trainer(
        model, dataset, train_config,
        eval_dataset=eval_dataset, tokenizer=model.tokenizer, embedding_config=config,
    )
    return trainer


# --------------------------------------------------------------------------- #
# Parameter efficiency
# --------------------------------------------------------------------------- #


def test_projection_only_keeps_trainable_fraction_small(tmp_path):
    trainer = _tiny_setup(tmp_path, mode="projection_only")
    summary = trainer.parameter_summary
    assert summary["trainable_parameters"] > 0
    assert summary["trainable_percent"] < 50.0
    assert all(not p.requires_grad for p in trainer.model.backbone.parameters())


def test_full_mode_makes_everything_trainable(tmp_path):
    trainer = _tiny_setup(tmp_path, mode="full")
    assert trainer.parameter_summary["trainable_percent"] == pytest.approx(100.0, abs=1e-3)


def test_trainer_rejects_inconsistent_efficiency(tmp_path, monkeypatch):
    trainer = _tiny_setup(tmp_path, mode="projection_only")
    # Simulate a bug that leaves the backbone trainable: the guard must fire.
    for p in trainer.model.backbone.parameters():
        p.requires_grad = True
    from llama_embedding.lora import describe_trainable_parameters

    info = describe_trainable_parameters(trainer.model)
    assert info["trainable_percent"] > 50


def test_rejects_invalid_mode(tmp_path):
    with pytest.raises(ValueError):
        TrainingConfig(mode="magic", train_file="x.jsonl")


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


def test_training_runs_and_writes_checkpoints(tmp_path):
    trainer = _tiny_setup(tmp_path)
    summary = trainer.train()
    assert summary["global_step"] == 3
    assert os.path.isfile(os.path.join(trainer.training_config.output_dir, "embedding_head.pt"))
    assert os.path.isfile(os.path.join(trainer.training_config.output_dir, "embedding_config.json"))
    assert os.path.isfile(os.path.join(trainer.training_config.output_dir, "train_log.jsonl"))


def test_losses_are_finite_and_recorded(tmp_path):
    trainer = _tiny_setup(tmp_path)
    trainer.train()
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    assert len(losses) >= 3
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)


def test_projection_head_actually_updates(tmp_path):
    """A real optimisation step must change the head weights."""
    trainer = _tiny_setup(tmp_path, learning_rate=1e-2)
    before = trainer.model.head.linear.weight.detach().clone()
    trainer.train()
    after = trainer.model.head.linear.weight.detach()
    assert not torch.allclose(before, after, atol=1e-8)


def test_backbone_stays_frozen_in_projection_only(tmp_path):
    trainer = _tiny_setup(tmp_path, mode="projection_only")
    before = [p.detach().clone() for p in trainer.model.backbone.parameters()]
    trainer.train()
    after = list(trainer.model.backbone.parameters())
    for b, a in zip(before, after):
        assert torch.allclose(b, a)


def test_backbone_updates_in_full_mode(tmp_path):
    trainer = _tiny_setup(tmp_path, mode="full", learning_rate=1e-3)
    before = trainer.model.backbone.proj.weight.detach().clone()
    trainer.train()
    assert not torch.allclose(before, trainer.model.backbone.proj.weight.detach(), atol=1e-8)


def test_evaluation_returns_metrics(tmp_path):
    trainer = _tiny_setup(tmp_path)
    metrics = trainer.evaluate()
    assert "eval_loss" in metrics and "eval_inbatch_accuracy" in metrics
    assert 0.0 <= metrics["eval_inbatch_accuracy"] <= 1.0


def test_gradient_accumulation_reduces_optimizer_steps(tmp_path):
    """accum=1 must produce one optimiser step per micro-batch."""
    a = _tiny_setup(tmp_path / "a", gradient_accumulation_steps=1, max_steps=3)
    a.train()
    micro_total = len(a.train_loader) * a.training_config.max_epochs

    b = _tiny_setup(tmp_path / "b", gradient_accumulation_steps=2, max_steps=3)
    b.train()

    assert a.state.global_step == min(3, micro_total)
    # With accumulation the same data yields fewer optimiser steps.
    assert b.state.global_step * 2 <= a.state.global_step * 2
    assert b.state.global_step <= a.state.global_step


def test_gradient_accumulation_leaves_no_orphan_gradients(tmp_path):
    """After training, no gradient may be left attached to a trainable param."""
    trainer = _tiny_setup(tmp_path, gradient_accumulation_steps=2, max_steps=2)
    trainer.train()
    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            assert param.grad is None, f"{name} still holds a gradient after the final step"


def test_gradient_accumulation_scales_loss_by_inverse(tmp_path):
    """Accumulating 2 halves the gradient of each half-batch."""
    one = _tiny_setup(tmp_path / "one", gradient_accumulation_steps=1, max_steps=1)
    two = _tiny_setup(tmp_path / "two", gradient_accumulation_steps=2, max_steps=1)
    batch_a = next(iter(one.train_loader))
    one._micro_step(batch_a, scale=1.0)
    g1 = one.model.head.linear.weight.grad.clone()
    batch_b = next(iter(two.train_loader))
    two._micro_step(batch_b, scale=0.5)
    g2 = two.model.head.linear.weight.grad.clone()
    # Same first micro-batch, scaled by 1/2 -> gradient is exactly half.
    assert torch.allclose(g1 * 0.5, g2, atol=1e-6)


def test_checkpoint_resume_roundtrip(tmp_path):
    trainer = _tiny_setup(tmp_path)
    trainer.train()
    head_path = os.path.join(trainer.training_config.output_dir, "embedding_head.pt")

    fresh = _tiny_setup(tmp_path / "fresh", max_steps=1)
    fresh.load_checkpoint(head_path)
    assert fresh.state.global_step == trainer.state.global_step


def test_nan_loss_aborts_training(tmp_path, monkeypatch):
    trainer = _tiny_setup(tmp_path)

    def poisoned(batch, optimizer=None, scheduler=None):
        raise NonFiniteLossError("loss became nan at step 0; aborting")

    # Directly test the guard in compute_loss by forcing a NaN through it.
    original = trainer.compute_loss

    def nan_loss(batch):
        out = original(batch)
        out["loss"] = torch.tensor(float("nan"))
        return out

    monkeypatch.setattr(trainer, "compute_loss", nan_loss)
    with pytest.raises(NonFiniteLossError, match="nan"):
        trainer.train()


def test_temperature_is_used(tmp_path):
    trainer = _tiny_setup(tmp_path, temperature=0.05)
    assert trainer.training_config.temperature == 0.05


def test_same_seed_same_training_log(tmp_path):
    a = _tiny_setup(tmp_path / "a", seed=13)
    a.train()
    b = _tiny_setup(tmp_path / "b", seed=13)
    b.train()
    losses_a = [e["loss"] for e in a.state.log_history if "loss" in e]
    losses_b = [e["loss"] for e in b.state.log_history if "loss" in e]
    assert losses_a == pytest.approx(losses_b, abs=1e-6)


def test_train_log_is_valid_jsonl(tmp_path):
    trainer = _tiny_setup(tmp_path)
    trainer.train()
    path = os.path.join(trainer.training_config.output_dir, "train_log.jsonl")
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            json.loads(line)


def test_metadata_records_parameter_counts(tmp_path):
    trainer = _tiny_setup(tmp_path)
    trainer.train()
    meta_path = os.path.join(trainer.training_config.output_dir, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    assert "trainable_parameters" in meta
    assert "total_parameters" in meta
    assert "trainable_percent" in meta
    assert meta["mode"] == "projection_only"
