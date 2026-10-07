"""Pooling unit tests: mask awareness, padding exclusion, shape contracts.

The central guarantee under test: **padding tokens never influence the sentence
vector.** That is asserted by mutating the padded positions to wild values and
checking the pooled output is bit-identical.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from llama_embedding.pooling import (
    LastTokenPooling,
    MeanPooling,
    PoolingStrategy,
    WeightedMeanPooling,
    build_pooling,
    get_pooling,
)


# --------------------------------------------------------------------------- #
# Shape contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strategy_name", ["mean", "last_token", "weighted_mean"])
def test_output_shape_is_b_by_h(hidden_states, attention_mask, strategy_name):
    pooling = get_pooling(strategy_name)
    out = pooling(hidden_states, attention_mask)
    batch, _, hidden = hidden_states.shape
    assert out.shape == (batch, hidden)


@pytest.mark.parametrize("strategy_name", ["mean", "last_token", "weighted_mean"])
def test_all_strategies_subclass_base(strategy_name):
    assert isinstance(get_pooling(strategy_name), PoolingStrategy)


# --------------------------------------------------------------------------- #
# Padding exclusion — the key property
# --------------------------------------------------------------------------- #


def test_mean_pooling_ignores_padding(hidden_states, attention_mask):
    """Garbage written into padded slots must not change the pooled vector."""
    pooling = MeanPooling()
    baseline = pooling(hidden_states, attention_mask)

    poisoned = hidden_states.clone()
    poisoned[1, 4:] = 1e4   # row 1 is padded from position 4
    poisoned[2, 2:] = -1e4  # row 2 is padded from position 2
    poisoned[3, 1:] = 3.5e3
    after = pooling(poisoned, attention_mask)

    assert torch.allclose(baseline, after, atol=1e-6)


def test_last_token_pooling_ignores_padding(hidden_states, attention_mask):
    pooling = LastTokenPooling()
    baseline = pooling(hidden_states, attention_mask)

    poisoned = hidden_states.clone()
    poisoned[1, 4:] = 1e4
    poisoned[2, 2:] = -1e4
    after = pooling(poisoned, attention_mask)
    assert torch.allclose(baseline, after, atol=1e-6)


def test_weighted_mean_pooling_ignores_padding(hidden_states, attention_mask):
    weights = torch.linspace(0.1, 1.0, hidden_states.shape[1]).unsqueeze(0).repeat(4, 1)
    pooling = WeightedMeanPooling()
    baseline = pooling(hidden_states, attention_mask, weights=weights)

    poisoned = hidden_states.clone()
    poisoned[1, 4:] = 1e4
    poisoned[2, 2:] = -1e4
    after = pooling(poisoned, attention_mask, weights=weights)
    assert torch.allclose(baseline, after, atol=1e-6)


# --------------------------------------------------------------------------- #
# Numerical correctness
# --------------------------------------------------------------------------- #


def test_mean_pooling_matches_manual_computation(hidden_states, attention_mask):
    pooling = MeanPooling()
    out = pooling(hidden_states, attention_mask)

    for b in range(hidden_states.shape[0]):
        length = int(attention_mask[b].sum())
        manual = hidden_states[b, :length].mean(dim=0)
        assert torch.allclose(out[b], manual, atol=1e-6)


def test_mean_pooling_on_single_token_sequence():
    hidden = torch.tensor([[[3.0, 4.0]]])  # one token
    mask = torch.tensor([[1]])
    out = MeanPooling()(hidden, mask)
    assert torch.allclose(out, torch.tensor([[3.0, 4.0]]))


def test_last_token_pooling_picks_final_real_token(hidden_states, attention_mask):
    pooling = LastTokenPooling()
    out = pooling(hidden_states, attention_mask)

    for b in range(hidden_states.shape[0]):
        length = int(attention_mask[b].sum())
        assert torch.allclose(out[b], hidden_states[b, length - 1], atol=1e-6)


def test_last_token_pooling_with_left_padding():
    hidden = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    left_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    out = LastTokenPooling()(hidden, left_mask)
    # Row 0: last active index is 2 -> hidden[0, 2]
    assert torch.allclose(out[0], hidden[0, 2])
    assert torch.allclose(out[1], hidden[1, 2])


def test_weighted_mean_pooling_equals_masked_weighted_average(hidden_states, attention_mask):
    weights = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 2.0, 3.0, 4.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    out = WeightedMeanPooling()(hidden_states, attention_mask, weights=weights)

    expected_row1 = (hidden_states[1, 0] * 1 + hidden_states[1, 1] * 2 +
                     hidden_states[1, 2] * 3 + hidden_states[1, 3] * 4) / 10.0
    assert torch.allclose(out[1], expected_row1, atol=1e-6)


def test_weighted_mean_falls_back_to_plain_mean_without_weights(hidden_states, attention_mask):
    weighted = WeightedMeanPooling()(hidden_states, attention_mask)
    plain = MeanPooling()(hidden_states, attention_mask)
    assert torch.allclose(weighted, plain, atol=1e-6)


def test_pooling_is_differentiable(hidden_states, attention_mask):
    hidden = hidden_states.clone().requires_grad_(True)
    MeanPooling()(hidden, attention_mask).sum().backward()
    assert hidden.grad is not None
    # No gradient may flow into padded positions.
    grad = hidden.grad
    assert grad[3, 1:].abs().max() == 0.0


# --------------------------------------------------------------------------- #
# Validation / error handling
# --------------------------------------------------------------------------- #


def test_rejects_all_padding_row(hidden_states):
    mask = torch.zeros(4, 6, dtype=torch.long)
    with pytest.raises(ValueError, match="no active tokens"):
        MeanPooling()(hidden_states, mask)


def test_rejects_shape_mismatched_mask(hidden_states):
    mask = torch.ones(4, 5, dtype=torch.long)
    with pytest.raises(ValueError, match="does not match"):
        MeanPooling()(hidden_states, mask)


def test_rejects_2d_input():
    with pytest.raises(ValueError, match=r"\[B, T, H\]"):
        MeanPooling()(torch.randn(3, 5), torch.ones(3, 5, dtype=torch.long))


def test_rejects_missing_mask(hidden_states):
    with pytest.raises(ValueError, match="attention_mask is required"):
        MeanPooling()(hidden_states, None)


def test_weighted_rejects_mismatched_weights(hidden_states, attention_mask):
    with pytest.raises(ValueError, match=r"\[B, T\]"):
        WeightedMeanPooling()(hidden_states, attention_mask, weights=torch.ones(4, 3))


def test_weighted_rejects_all_zero_weights(hidden_states, attention_mask):
    with pytest.raises(ValueError, match="all weights are zero"):
        WeightedMeanPooling()(hidden_states, attention_mask, weights=torch.zeros(4, 6))


def test_accepts_bool_mask(hidden_states, attention_mask):
    bool_mask = attention_mask.bool()
    a = MeanPooling()(hidden_states, attention_mask)
    b = MeanPooling()(hidden_states, bool_mask)
    assert torch.allclose(a, b)


def test_unknown_pooling_name_raises():
    with pytest.raises(KeyError):
        get_pooling("attention")
    with pytest.raises(KeyError):
        get_pooling("")


def test_build_pooling_from_config(tiny_config):
    tiny_config.pooling = "last_token"
    pooling = build_pooling(tiny_config)
    assert isinstance(pooling, LastTokenPooling)


def test_build_pooling_rejects_bad_config_object():
    class Bad:
        pooling = "nonsense"

    with pytest.raises(KeyError):
        build_pooling(Bad())


def test_padding_token_values_cannot_change_mean(hidden_states, attention_mask):
    """Explicitly set padding to the same constant: result must be invariant."""
    a = hidden_states.clone()
    a[3, 1:] = 42.0
    b = hidden_states.clone()
    b[3, 1:] = -17.0
    assert torch.allclose(MeanPooling()(a, attention_mask), MeanPooling()(b, attention_mask))
