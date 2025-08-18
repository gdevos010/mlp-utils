# ABOUTME: Stress tests for Tversky similarity under extreme and degenerate cases.
# ABOUTME: Covers zeros, disjoint supports, large magnitudes, smoothing, and thresholds.

import torch
import torch.nn.functional as F

from mlp_utils.layers import tversky_similarity


def test_all_zeros_inputs() -> None:
    x = torch.zeros(5, 7)
    y = torch.zeros(5, 7)
    s = tversky_similarity(x, y)
    assert torch.isfinite(s).all()
    assert (s >= 0).all() and (s <= 1 + 1e-7).all()


def test_disjoint_supports_positive() -> None:
    torch.manual_seed(0)
    a = F.relu(torch.randn(6, 9)) + 5.0
    b = F.relu(torch.randn(6, 9))
    s = tversky_similarity(a, b)
    assert torch.isfinite(s).all()
    assert (s >= 0).all() and (s <= 1 + 1e-7).all()


def test_large_magnitudes_stability() -> None:
    torch.manual_seed(1)
    x = torch.rand(4, 1024) * 1e8
    y = torch.rand(4, 1024) * 1e8
    s = tversky_similarity(x, y, input_transform="relu")
    assert torch.isfinite(s).all()
    assert (s >= -1e-7).all() and (s <= 1 + 1e-7).all()


def test_smoothing_tau_small_and_large() -> None:
    torch.manual_seed(2)
    x = torch.rand(3, 64)
    y = torch.rand(3, 64)
    s_small = tversky_similarity(x, y, smoothing_tau=1e-4)
    s_large = tversky_similarity(x, y, smoothing_tau=1.0)
    for s in (s_small, s_large):
        assert torch.isfinite(s).all()
        assert (s >= -1e-7).all() and (s <= 1 + 1e-7).all()


def test_ignorematch_and_match_threshold_effects() -> None:
    # With high threshold and clamp01 inputs near 1, fewer positions count as "both present"
    # Expect difference contributions to be less zeroed when threshold is very high vs zero.
    torch.manual_seed(3)
    x = torch.rand(5, 16)
    y = torch.rand(5, 16)
    s_thr0 = tversky_similarity(
        x, y, input_transform="clamp01", difference_reduction="ignorematch", match_threshold=0.0
    )
    s_thr_hi = tversky_similarity(
        x, y, input_transform="clamp01", difference_reduction="ignorematch", match_threshold=0.99
    )
    # Both within bounds and finite
    assert torch.isfinite(s_thr0).all() and torch.isfinite(s_thr_hi).all()
    assert (s_thr0 >= -1e-7).all() and (s_thr0 <= 1 + 1e-7).all()
    assert (s_thr_hi >= -1e-7).all() and (s_thr_hi <= 1 + 1e-7).all()
