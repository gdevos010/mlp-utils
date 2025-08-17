# ABOUTME: Placeholder tests for Tversky similarity and projection layer (scaffold).
# ABOUTME: Entire module is skipped until implementations are added in later steps.
import math
from typing import Callable

import pytest
import torch

from mlp_utils.layers import tversky_similarity


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
