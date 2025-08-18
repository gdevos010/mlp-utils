# ABOUTME: AMP/autocast smoke tests for Tversky modules on CPU (bf16 if available).
# ABOUTME: Ensures forward/backward work under autocast without errors.

import pytest
import torch

from mlp_utils.layers import TverskyProjection, TverskyFeatureSharing


@pytest.mark.skipif(
    not torch.cuda.is_available() and not torch.backends.cpu.has_bfloat16,
    reason="no AMP datatype available (CUDA or CPU bf16)",
)
def test_tversky_projection_autocast_smoke() -> None:
    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    proj = TverskyProjection(16, 8, input_transform="clamp01").to(device)
    x = torch.rand(4, 16, device=device, requires_grad=True)

    dtype = torch.bfloat16 if not use_cuda else torch.float16
    autocast = torch.cuda.amp.autocast if use_cuda else torch.cpu.amp.autocast

    with autocast(dtype=dtype):
        y = proj(x)
        loss = y.sum()
    loss.backward()
    assert y.shape == (4, 8)
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.skipif(
    not torch.cuda.is_available() and not torch.backends.cpu.has_bfloat16,
    reason="no AMP datatype available (CUDA or CPU bf16)",
)
def test_tversky_feature_sharing_autocast_smoke() -> None:
    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    fs = TverskyFeatureSharing(
        input_dim=12,
        num_features=10,
        output_dim=6,
        s1_input_transform="clamp01",
        s2_input_transform="clamp01",
    ).to(device)
    x = torch.rand(3, 12, device=device, requires_grad=True)

    dtype = torch.bfloat16 if not use_cuda else torch.float16
    autocast = torch.cuda.amp.autocast if use_cuda else torch.cpu.amp.autocast

    with autocast(dtype=dtype):
        y = fs(x)
        loss = y.sum()
    loss.backward()
    assert y.shape == (3, 6)
    assert x.grad is not None and torch.isfinite(x.grad).all()
