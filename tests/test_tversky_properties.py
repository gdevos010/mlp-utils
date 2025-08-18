# ABOUTME: Property-based tests for Tversky similarity and utilities.
# ABOUTME: Covers bounds, symmetry swap (α/β), monotonicity, broadcasting, precision, and product reduction.

import math
import pytest
import numpy as np
import torch

hypo = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # type: ignore  # noqa: E402
from hypothesis.extra.numpy import arrays as hnp_arrays  # type: ignore  # noqa: E402

from mlp_utils.layers import pairwise_tversky, tversky_similarity


def _to_torch(array: np.ndarray, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32 if dtype == torch.float32 else np.float64)).to(dtype)


@settings(deadline=None, max_examples=60)
@given(
    D=st.integers(min_value=1, max_value=32),
    transform=st.sampled_from([None, "relu", "clamp01", "sigmoid"]),
)
def test_bounds_property(D: int, transform: str | None) -> None:
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(size=(D,)).astype(np.float64)
    y_np = rng.standard_normal(size=(D,)).astype(np.float64)
    x = _to_torch(x_np, dtype=torch.float64)
    y = _to_torch(y_np, dtype=torch.float64)
    s = tversky_similarity(x, y, input_transform=transform)
    assert s.ndim == 0
    s_val = float(s.item())
    assert s_val >= -1e-7 and s_val <= 1.0 + 1e-7


@settings(deadline=None, max_examples=60)
@given(D=st.integers(min_value=1, max_value=32), a=st.floats(0.05, 0.95), b=st.floats(0.05, 0.95))
def test_swap_alpha_beta_symmetry(D: int, a: float, b: float) -> None:
    # Property: s(x, y; α, β) == s(y, x; β, α)
    rng = np.random.default_rng(1)
    x = _to_torch(rng.random(size=(D,)), dtype=torch.float64)
    y = _to_torch(rng.random(size=(D,)), dtype=torch.float64)
    s_xy = tversky_similarity(x, y, alpha=a, beta=b)
    s_yx = tversky_similarity(y, x, alpha=b, beta=a)
    assert math.isclose(float(s_xy.item()), float(s_yx.item()), rel_tol=1e-6, abs_tol=1e-6)


@settings(deadline=None, max_examples=50)
@given(D=st.integers(min_value=1, max_value=32))
def test_monotonicity_in_shared_mass_property(D: int) -> None:
    rng = np.random.default_rng(2)
    y = _to_torch(rng.random(size=(D,)))  # nonnegative
    x1 = _to_torch(rng.random(size=(D,)))  # nonnegative
    delta = 0.5 * torch.relu(y - x1)
    x2 = x1 + delta
    s1 = tversky_similarity(x1, y)
    s2 = tversky_similarity(x2, y)
    assert float(s2.item()) >= float(s1.item()) - 1e-7


@settings(deadline=None, max_examples=40)
@given(
    B=st.integers(min_value=1, max_value=3),
    T=st.integers(min_value=1, max_value=3),
    D=st.integers(min_value=1, max_value=8),
    K=st.integers(min_value=1, max_value=5),
)
def test_pairwise_broadcast_shapes_property(B: int, T: int, D: int, K: int) -> None:
    rng = np.random.default_rng(3)
    x = _to_torch(rng.random(size=(B, T, D)))
    P = _to_torch(rng.random(size=(K, D)))
    sims = pairwise_tversky(
        x, P, input_transform="clamp01", intersection_reduction="sum", difference_reduction="subtractmatch"
    )
    assert sims.shape == (B, T, K)


@settings(deadline=None, max_examples=30)
@given(D=st.integers(min_value=4, max_value=64))
def test_product_reduction_finiteness(D: int) -> None:
    rng = np.random.default_rng(4)
    # Keep components small but positive to exercise product path
    x = _to_torch(1e-3 * rng.random(size=(D,)))
    y = _to_torch(1e-3 * rng.random(size=(D,)))
    s = tversky_similarity(x, y, input_transform=None, nonnegative=True, intersection_reduction="product")
    assert torch.isfinite(s)
    s_val = float(s.item())
    assert s_val >= -1e-9 and s_val <= 1.0 + 1e-9


@settings(deadline=None, max_examples=40)
@given(D=st.integers(min_value=1, max_value=32))
def test_precision_parity_float32_vs_float64(D: int) -> None:
    rng = np.random.default_rng(5)
    x64 = _to_torch(rng.standard_normal(size=(D,)), dtype=torch.float64)
    y64 = _to_torch(rng.standard_normal(size=(D,)), dtype=torch.float64)
    s64 = tversky_similarity(x64, y64)
    s32 = tversky_similarity(x64.float(), y64.float()).double()
    assert torch.allclose(s64, s32, rtol=1e-5, atol=1e-6)
