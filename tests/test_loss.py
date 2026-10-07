"""Contrastive loss tests: InfoNCE correctness, temperature, hard negatives."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from llama_embedding.config import DEFAULT_TEMPERATURE
from llama_embedding.losses import (
    build_matryoshka_prefixes,
    contrastive_loss,
    info_nce,
    matryoshka_info_nce,
    pairwise_info_nce,
)
from llama_embedding.projection import l2_normalize


# --------------------------------------------------------------------------- #
# InfoNCE
# --------------------------------------------------------------------------- #


def test_info_nce_perfect_alignment_is_near_zero():
    """When queries == documents, the diagonal dominates and loss -> 0."""
    torch.manual_seed(0)
    x = l2_normalize(torch.randn(32, 64))
    loss = info_nce(x, x, temperature=DEFAULT_TEMPERATURE)
    assert loss.item() < 1e-3


def test_info_nce_worst_case_is_log_batch_size():
    """Anti-correlated pairs: with logits=0 everywhere the loss is log(B)."""
    torch.manual_seed(1)
    q = l2_normalize(torch.randn(8, 16))
    d = -q  # every off-diagonal scores high, diagonal scores -1
    loss = info_nce(q, d, temperature=1e6)  # near-uniform logits
    assert abs(loss.item() - math.log(8)) < 0.15


def test_info_nce_matches_manual_cross_entropy():
    """Reproduce the definition by hand."""
    torch.manual_seed(2)
    q = l2_normalize(torch.randn(4, 8))
    d = l2_normalize(torch.randn(4, 8))
    tau = 0.05
    logits = (q @ d.T) / tau
    target = torch.arange(4)
    manual = torch.nn.functional.cross_entropy(logits, target)
    assert torch.allclose(info_nce(q, d, temperature=tau), manual, atol=1e-6)


def test_info_nce_decreases_as_alignment_improves():
    torch.manual_seed(3)
    base = l2_normalize(torch.randn(16, 32))
    noise = l2_normalize(torch.randn(16, 32))
    good = info_nce(base, base + 0.1 * noise, temperature=0.05)
    bad = info_nce(base, base + 3.0 * noise, temperature=0.05)
    assert good.item() < bad.item()


def test_info_nce_is_scale_invariant():
    """Cosine + normalisation: scaling inputs must not change the loss."""
    torch.manual_seed(4)
    q = torch.randn(6, 16)
    d = torch.randn(6, 16)
    a = info_nce(q, d, temperature=0.05)
    b = info_nce(q * 10.0, d * 0.05, temperature=0.05)
    assert torch.allclose(a, b, atol=1e-5)


def test_info_nce_symmetric_averages_both_directions():
    torch.manual_seed(5)
    q = l2_normalize(torch.randn(8, 16))
    d = l2_normalize(torch.randn(8, 16))
    forward = info_nce(q, d, symmetric=False)
    both = info_nce(q, d, symmetric=True)
    assert not torch.allclose(forward, both)


def test_info_nce_temperature_changes_loss():
    torch.manual_seed(6)
    q = l2_normalize(torch.randn(8, 16))
    d = l2_normalize(torch.randn(8, 16))
    hot = info_nce(q, d, temperature=0.01)
    cold = info_nce(q, d, temperature=1.0)
    assert hot.item() != pytest.approx(cold.item(), abs=1e-6)


def test_info_nce_rejects_temperature_zero():
    q, d = torch.randn(2, 4), torch.randn(2, 4)
    with pytest.raises(ValueError, match="temperature must be > 0"):
        info_nce(q, d, temperature=0.0)


def test_info_nce_rejects_batch_of_one():
    with pytest.raises(ValueError, match="batch size of at least 2"):
        info_nce(torch.randn(1, 4), torch.randn(1, 4))


def test_info_nce_rejects_batch_mismatch():
    with pytest.raises(ValueError, match="batch size mismatch"):
        info_nce(torch.randn(3, 4), torch.randn(4, 4))


def test_info_nce_rejects_dim_mismatch():
    with pytest.raises(ValueError, match="embedding dim mismatch"):
        info_nce(torch.randn(3, 4), torch.randn(3, 8))


def test_info_nce_label_smoothing_flattens_the_distribution():
    """Smoothing spreads mass off the diagonal, so it cannot exceed the hard loss.

    Cross-entropy against a uniform target is a mixture of the one-hot loss and a
    constant, so smoothing always *reduces* (or leaves) the value. The earlier
    expectation that it increases the loss was simply wrong.
    """
    torch.manual_seed(8)
    q = l2_normalize(torch.randn(6, 16))
    d = l2_normalize(torch.randn(6, 16))
    hard = info_nce(q, d).item()
    smooth = info_nce(q, d, label_smoothing=0.2).item()
    assert smooth <= hard + 1e-6


def test_info_nce_label_smoothing_reduces_confidence_on_a_wrong_prediction():
    """Where it does raise the loss is a badly wrong prediction."""
    torch.manual_seed(81)
    q = l2_normalize(torch.randn(8, 16))
    d = -q  # diagonal is the worst possible match
    hard = info_nce(q, d).item()
    smooth = info_nce(q, d, label_smoothing=0.2).item()
    assert smooth > hard


def test_info_nce_rejects_invalid_label_smoothing():
    q, d = torch.randn(3, 4), torch.randn(3, 4)
    with pytest.raises(ValueError, match="label_smoothing"):
        info_nce(q, d, label_smoothing=1.0)


# --------------------------------------------------------------------------- #
# Hard negatives
# --------------------------------------------------------------------------- #


def test_pairwise_info_nce_lower_when_positive_is_clearly_separated():
    """A positive far from every negative must score better than a confusing one.

    Note the direction: a negative that sits *close* to the positive is HARD and
    yields a HIGH loss, so this asserts the opposite of an intuition trap.
    """
    torch.manual_seed(9)
    q = l2_normalize(torch.randn(6, 16))
    p = l2_normalize(torch.randn(6, 16))
    hard_neg = l2_normalize(p + 0.05 * torch.randn(6, 16))   # nearly identical
    easy_neg = l2_normalize(torch.randn(6, 16))              # clearly different
    hard = pairwise_info_nce(q, p, hard_neg, temperature=0.05)
    easy = pairwise_info_nce(q, p, easy_neg, temperature=0.05)
    assert hard.item() > easy.item()


def test_pairwise_info_nce_zero_when_positive_perfectly_separated():
    """pos = +1, neg = -1 exactly -> the objective reaches its floor."""
    q = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    p = q.clone()
    n = -q.clone()
    loss = pairwise_info_nce(q, p, n, temperature=0.05)
    assert loss.item() < 1e-4


def test_pairwise_info_nce_is_max_over_negatives():
    """Adding a harder negative must not decrease the loss."""
    torch.manual_seed(30)
    q = l2_normalize(torch.randn(4, 16))
    p = l2_normalize(torch.randn(4, 16))
    far = l2_normalize(torch.randn(4, 1, 16))
    both = l2_normalize(torch.cat([far, p.unsqueeze(1) + 0.02 * torch.randn(4, 1, 16)], dim=1))
    assert pairwise_info_nce(q, p, both, temperature=0.05).item() >= \
        pairwise_info_nce(q, p, far, temperature=0.05).item() - 1e-5


def test_pairwise_info_nce_accepts_multiple_negatives():
    torch.manual_seed(10)
    q = l2_normalize(torch.randn(4, 16))
    p = l2_normalize(torch.randn(4, 16))
    negatives = l2_normalize(torch.randn(4, 3, 16))
    loss = pairwise_info_nce(q, p, negatives, temperature=0.05)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_pairwise_info_nce_rejects_bad_negative_shape():
    q, p = torch.randn(3, 8), torch.randn(3, 8)
    with pytest.raises(ValueError, match=r"\[B, D\] or \[B, K, D\]"):
        pairwise_info_nce(q, p, torch.randn(3, 2, 2, 8))


def test_pairwise_info_nce_rejects_batch_mismatch():
    q, p = torch.randn(3, 8), torch.randn(3, 8)
    with pytest.raises(ValueError, match="batch size must match"):
        pairwise_info_nce(q, p, torch.randn(4, 8))


# --------------------------------------------------------------------------- #
# Matryoshka
# --------------------------------------------------------------------------- #


def test_build_matryoshka_prefixes_sorted_and_capped():
    assert build_matryoshka_prefixes(768, (256, 1024, 384)) == (256, 384, 768)


def test_matryoshka_info_nce_returns_loss_and_dims():
    torch.manual_seed(12)
    q = l2_normalize(torch.randn(6, 1024))
    d = l2_normalize(torch.randn(6, 1024))
    loss, dims = matryoshka_info_nce(q, d, (256, 512), temperature=0.05)
    assert dims == (256, 512, 1024)
    assert torch.isfinite(loss)


def test_contrastive_loss_with_matryoshka_dims():
    torch.manual_seed(13)
    q = l2_normalize(torch.randn(4, 768))
    d = l2_normalize(torch.randn(4, 768))
    out = contrastive_loss(q, d, matryoshka_dims=(256, 384))
    assert "matryoshka" in out and "loss" in out
    assert torch.isfinite(out["loss"])


# --------------------------------------------------------------------------- #
# Composed objective
# --------------------------------------------------------------------------- #


def test_contrastive_loss_keys_and_finiteness():
    torch.manual_seed(14)
    q = l2_normalize(torch.randn(5, 768))
    d = l2_normalize(torch.randn(5, 768))
    out = contrastive_loss(q, d)
    assert "in_batch" in out and "loss" in out
    assert torch.isfinite(out["loss"])


def test_contrastive_loss_includes_hard_negatives():
    torch.manual_seed(15)
    q = l2_normalize(torch.randn(5, 768))
    d = l2_normalize(torch.randn(5, 768))
    neg = l2_normalize(torch.randn(5, 2, 768))
    out = contrastive_loss(q, d, negatives=neg, hard_negative_weight=0.5)
    assert "hard_negative" in out
    assert torch.isfinite(out["loss"])


def test_contrastive_loss_is_differentiable():
    torch.manual_seed(16)
    q = l2_normalize(torch.randn(4, 768)).requires_grad_(True)
    d = l2_normalize(torch.randn(4, 768))
    out = contrastive_loss(q, d)
    out["loss"].backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_default_temperature_is_documented_value():
    """Guards the documented default so it cannot drift silently."""
    assert DEFAULT_TEMPERATURE == 0.02


def test_contrastive_loss_over_batch_is_finite_and_positive():
    torch.manual_seed(17)
    for batch in (2, 8, 32):
        q = l2_normalize(torch.randn(batch, 768))
        d = l2_normalize(torch.randn(batch, 768))
        loss = contrastive_loss(q, d)["loss"]
        assert loss.item() > 0.0 and torch.isfinite(loss)
