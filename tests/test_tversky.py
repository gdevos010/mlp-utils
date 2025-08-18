# ABOUTME: Placeholder tests for Tversky similarity and projection layer (scaffold).
# ABOUTME: Entire module is skipped until implementations are added in later steps.
import math
from typing import Callable

import pytest
import torch
import torch.nn.functional as F

from mlp_utils.layers import (
    TverskyProjection,
    TverskyFeatureSharing,
    pairwise_tversky,
    tversky_similarity,
    tversky_attributions,
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


def test_monotonicity_in_shared_mass() -> None:
    torch.manual_seed(9)
    D = 10
    y = F.relu(torch.randn(D))
    x1 = F.relu(torch.randn(D))
    # Increase overlap only where x1 <= y so A_only does not increase
    delta = 0.5 * torch.relu(y - x1)
    x2 = x1 + delta  # ensures x2 <= x1 + (y - x1) = y
    s1 = tversky_similarity(x1, y)
    s2 = tversky_similarity(x2, y)
    assert s2.item() >= s1.item() - 1e-7


def test_sequential_integration_forward_backward() -> None:
    torch.manual_seed(15)
    in_dim, out_dim = 6, 3
    model = torch.nn.Sequential(
        TverskyProjection(in_dim, out_dim),
        torch.nn.Softmax(dim=-1),
    )
    x = torch.rand(5, in_dim, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    proj: TverskyProjection = model[0]  # type: ignore[assignment]
    assert proj.weight.grad is not None and torch.isfinite(proj.weight.grad).all()


def test_feature_sharing_shapes_and_gradients() -> None:
    torch.manual_seed(21)
    B, D_in, M, K = 5, 6, 7, 3
    model = TverskyFeatureSharing(
        input_dim=D_in,
        num_features=M,
        output_dim=K,
        s1_input_transform="clamp01",
        s2_input_transform="clamp01",
    )
    x = torch.rand(B, D_in, requires_grad=True)
    y = model(x)
    assert y.shape == (B, K)
    loss = y.sum()
    loss.backward()
    # Gradients should flow to inputs and both stages
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert model.stage1.weight.grad is not None and torch.isfinite(model.stage1.weight.grad).all()
    assert model.stage2.weight.grad is not None and torch.isfinite(model.stage2.weight.grad).all()


def test_feature_sharing_membership_argmax_matches_identity_head() -> None:
    torch.manual_seed(22)
    D_in = 4
    M = 4
    K = 4
    fs = TverskyFeatureSharing(
        input_dim=D_in,
        num_features=M,
        output_dim=K,
        s1_input_transform="clamp01",
        s2_input_transform="clamp01",
        s2_alpha=1.0,
        s2_beta=0.0,
        s2_bias=False,
    )
    with torch.no_grad():
        fs.stage2.weight.copy_(torch.eye(M))
    x = torch.rand(8, D_in)
    z = fs.get_feature_memberships(x)
    y = fs(x)
    assert torch.equal(y.argmax(dim=-1), z.argmax(dim=-1))


def test_feature_sharing_output_bounds_nonnegative_inputs() -> None:
    torch.manual_seed(23)
    B, D_in, M, K = 4, 5, 6, 3
    fs = TverskyFeatureSharing(
        input_dim=D_in,
        num_features=M,
        output_dim=K,
        s1_input_transform="clamp01",
        s2_input_transform="clamp01",
        s1_bias=False,
        s2_bias=False,
    )
    x = torch.rand(B, D_in)
    y = fs(x)
    assert y.min().item() >= -1e-7
    assert y.max().item() <= 1.0 + 1e-7


def test_feature_sharing_jit_trace() -> None:
    torch.manual_seed(24)
    fs = TverskyFeatureSharing(
        input_dim=4,
        num_features=5,
        output_dim=3,
        s1_input_transform="clamp01",
        s2_input_transform="clamp01",
    )
    x = torch.rand(2, 4)
    traced = torch.jit.trace(fs, x)
    y_traced = traced(x)
    y_eager = fs(x)
    assert torch.allclose(y_traced, y_eager, rtol=1e-6, atol=1e-6)


def test_smoothing_tau_approaches_hard_ops() -> None:
    torch.manual_seed(10)
    x = torch.rand(6, 13)
    y = torch.rand(6, 13)
    s_hard = tversky_similarity(x, y)
    s_soft = tversky_similarity(x, y, smoothing_tau=1e-3)
    assert torch.allclose(s_soft, s_hard, rtol=1e-3, atol=1e-4)


def test_numerical_stability_various_inputs() -> None:
    zero = torch.zeros(5, 7)
    disjoint_a = F.relu(torch.randn(5, 7))
    disjoint_b = F.relu(torch.randn(5, 7)) + 5.0  # likely disjoint mass
    large = torch.rand(5, 7) * 1e6
    pairs = [
        (zero, zero),
        (disjoint_a, disjoint_b),
        (large, large * 0.5),
    ]
    for a, b in pairs:
        s = tversky_similarity(a, b)
        assert torch.isfinite(s).all()
        assert (s >= 0).all() and (s <= 1 + 1e-7).all()


@pytest.mark.slow
def test_xor_training_high_accuracy() -> None:
    torch.manual_seed(1234)
    X = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    Y = torch.tensor([0, 1, 1, 0])
    X = X.repeat(32, 1)
    Y = Y.repeat(32)

    model = torch.nn.Sequential(
        TverskyProjection(2, 2, input_transform="clamp01", nonnegative=True),
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.5)

    for _ in range(200):
        logits = model(X)
        loss = torch.nn.functional.cross_entropy(logits, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = model(X).argmax(dim=-1)
        acc = (preds == Y).float().mean().item()
    assert acc >= 0.9


@pytest.mark.slow
def test_gradcheck_tversky_similarity() -> None:
    torch.manual_seed(11)
    x = torch.rand(4, dtype=torch.float64, requires_grad=True) * 0.9 + 0.1
    y = torch.rand(4, dtype=torch.float64, requires_grad=True) * 0.9 + 0.1

    def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return tversky_similarity(a, b).sum()

    assert torch.autograd.gradcheck(fn, (x, y), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_attributions_shapes_and_sums() -> None:
    torch.manual_seed(12)
    x = torch.rand(3, 8)
    p = torch.rand(8)
    I_c, A_c, B_c = tversky_attributions(x, p)
    assert I_c.shape == x.shape
    assert A_c.shape == x.shape
    assert B_c.shape == x.shape
    # Consistency with internal aggregation
    I = I_c.sum(dim=-1)
    A = A_c.sum(dim=-1)
    B = B_c.sum(dim=-1)
    s = (I + 1e-6) / (I + 0.5 * A + 0.5 * B + 1e-6)
    s_ref = tversky_similarity(x, p)
    assert torch.allclose(s, s_ref, rtol=1e-5, atol=1e-6)


def test_attributions_smoothed_and_nonnegativity() -> None:
    torch.manual_seed(13)
    x = torch.randn(5, 6)
    p = torch.randn(6)
    I_c, A_c, B_c = tversky_attributions(x, p, input_transform="relu", smoothing_tau=0.2)
    assert torch.isfinite(I_c).all() and torch.isfinite(A_c).all() and torch.isfinite(B_c).all()
    assert (I_c >= 0).all() and (A_c >= 0).all() and (B_c >= 0).all()


def test_learnable_alpha_beta_and_normalization() -> None:
    torch.manual_seed(14)
    in_dim, out_dim = 6, 4
    # Learnable without normalization
    layer = TverskyProjection(in_dim, out_dim, learnable_alpha=True, learnable_beta=True)
    # Parameters should include unconstrained vars
    params = dict(layer.named_parameters())
    assert any("_alpha_unconstrained" in k for k in params)
    assert any("_beta_unconstrained" in k for k in params)

    x = torch.rand(3, in_dim)
    y = layer(x)
    assert y.shape == (3, out_dim)

    # With normalization, α+β ≈ 1
    layer2 = TverskyProjection(
        in_dim,
        out_dim,
        learnable_alpha=True,
        learnable_beta=True,
        alpha_beta_normalize=True,
    )
    with torch.no_grad():
        a_b_sum = layer2._compute_alpha_beta()[0] + layer2._compute_alpha_beta()[1]
    assert math.isclose(float(a_b_sum.item()), 1.0, rel_tol=1e-3, abs_tol=1e-3)
