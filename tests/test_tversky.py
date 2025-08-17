# ABOUTME: Placeholder tests for Tversky similarity and projection layer (scaffold).
# ABOUTME: Entire module is skipped until implementations are added in later steps.
import math
from typing import Callable

import pytest
import torch

from mlp_utils.layers import (
    TverskyProjection,
    pairwise_tversky,
    tversky_similarity,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_identity_similarity_is_one(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    x = torch.rand(3, 5, dtype=dtype)
    s = tversky_similarity(x, x)
    assert s.shape == (3,)
    assert torch.allclose(s, torch.ones_like(s), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "transform",
    [None, "relu", "clamp01", "sigmoid", lambda t: torch.clamp(t, min=0.0)],
)
def test_range_bounded_0_1(dtype: torch.dtype, transform: str | Callable) -> None:
    torch.manual_seed(1)
    x = torch.randn(7, 11, dtype=dtype)
    y = torch.randn(7, 11, dtype=dtype)
    s = tversky_similarity(x, y, input_transform=transform)
    assert s.min().item() >= 0.0 - 1e-7
    assert s.max().item() <= 1.0 + 1e-7


def test_asymmetry_alpha_ne_beta() -> None:
    a = torch.tensor([1.0, 2.0, 0.0])
    b = torch.tensor([0.0, 1.0, 3.0])
    s_ab = tversky_similarity(a, b, alpha=0.7, beta=0.3)
    s_ba = tversky_similarity(b, a, alpha=0.7, beta=0.3)
    assert not math.isclose(s_ab.item(), s_ba.item())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pairwise_shapes_and_loop_parity(dtype: torch.dtype) -> None:
    torch.manual_seed(2)
    B, D, K = 4, 7, 5
    x = torch.randn(B, D, dtype=dtype)
    P = torch.randn(K, D, dtype=dtype)
    sims = pairwise_tversky(x, P)
    assert sims.shape == (B, K)

    # Parity with explicit loop
    sims_loop = []
    for k in range(K):
        sims_loop.append(tversky_similarity(x, P[k]))
    sims_loop_t = torch.stack(sims_loop, dim=-1)
    assert torch.allclose(sims, sims_loop_t, rtol=1e-5, atol=1e-6)


def test_pairwise_broadcasting_time_axis() -> None:
    torch.manual_seed(3)
    B, T, D, K = 2, 3, 4, 6
    x = torch.randn(B, T, D)
    P = torch.randn(K, D)
    sims = pairwise_tversky(x, P)
    assert sims.shape == (B, T, K)


def test_pairwise_gradients_exist() -> None:
    torch.manual_seed(4)
    B, D, K = 3, 8, 4
    x = torch.randn(B, D, requires_grad=True)
    P = torch.randn(K, D, requires_grad=True)
    sims = pairwise_tversky(x, P)
    loss = sims.sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert P.grad is not None and torch.isfinite(P.grad).all()
    # Likely nonzero for random inputs
    assert x.grad.abs().sum().item() > 0
    assert P.grad.abs().sum().item() > 0


def test_tversky_projection_basic_forward_and_params() -> None:
    torch.manual_seed(5)
    in_dim, out_dim = 10, 7
    layer = TverskyProjection(in_dim, out_dim)
    # Parameter shapes
    assert layer.weight.shape == (out_dim, in_dim)
    assert layer.bias is None

    x = torch.randn(2, in_dim)
    y = layer(x)
    assert y.shape == (2, out_dim)

    # Range semantics for nonnegative inputs without bias
    x2 = torch.rand(4, in_dim)
    y2 = layer(x2)
    assert y2.min().item() >= 0.0 - 1e-7
    assert y2.max().item() <= 1.0 + 1e-7


def test_tversky_projection_bias_and_training_step() -> None:
    torch.manual_seed(6)
    in_dim, out_dim = 6, 3
    layer = TverskyProjection(in_dim, out_dim, bias=True)
    opt = torch.optim.SGD(layer.parameters(), lr=1e-1)
    x = torch.rand(16, in_dim)  # nonnegative to keep (0,1] semantics
    target = torch.zeros(16, out_dim)

    # Simple loss: mean squared error to zeros
    y0 = layer(x)
    loss0 = (y0 ** 2).mean()
    opt.zero_grad(set_to_none=True)
    loss0.backward()
    opt.step()

    y1 = layer(x)
    loss1 = (y1 ** 2).mean()
    # Expect some reduction (stochastic but should usually decrease a bit)
    assert loss1.item() <= loss0.item() + 1e-6 or loss1.item() < loss0.item()


def test_tversky_projection_state_dict_roundtrip() -> None:
    torch.manual_seed(7)
    in_dim, out_dim = 5, 4
    layer = TverskyProjection(in_dim, out_dim)
    x = torch.randn(3, in_dim)
    y_before = layer(x)

    state = layer.state_dict()
    layer2 = TverskyProjection(in_dim, out_dim)
    layer2.load_state_dict(state)
    y_after = layer2(x)
    assert torch.allclose(y_before, y_after, rtol=1e-6, atol=1e-7)


def test_jit_trace_forward() -> None:
    torch.manual_seed(8)
    in_dim, out_dim = 4, 3
    layer = TverskyProjection(in_dim, out_dim)
    x = torch.randn(2, in_dim)
    traced = torch.jit.trace(layer, x)
    y_traced = traced(x)
    y_eager = layer(x)
    assert torch.allclose(y_traced, y_eager, rtol=1e-6, atol=1e-6)
