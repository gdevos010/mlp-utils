"""Tests for the nGPT layer, including its integration with FastFeedForward."""
import torch
import torch.nn.functional as F

from mlp_utils.layers import NGPT, FastFeedForward


def test_ngpt_with_fastfeedforward_initialization() -> None:
    """Tests if nGPT can be initialized with a FastFeedForward network."""
    dim = 256
    depth = 3
    fff = FastFeedForward(dim=dim, depth=depth)
    ngpt_block = NGPT(feedforward_net=fff, dim=dim)

    assert ngpt_block is not None, "nGPT block should be initialized."
    assert (
        ngpt_block.feedforward_net is fff
    ), "Feedforward network should be set correctly."


def test_ngpt_with_fastfeedforward_forward_pass() -> None:
    """Tests the forward pass of nGPT wrapping a FastFeedForward network."""
    batch_size = 4
    seq_len = 16
    dim = 256
    depth = 3

    # 1. Create the FastFeedForward network
    fff = FastFeedForward(
        dim=dim,
        depth=depth,
        mult=4,
        glu_variant="swiglu",
    )

    # 2. Wrap it with nGPT
    ngpt_block = NGPT(
        feedforward_net=fff,
        dim=dim,
        scalar_alpha=False,
    )
    ngpt_block.eval()  # Ensure we're in eval mode

    # 3. Create a normalized input tensor
    h = torch.randn(batch_size, seq_len, dim)
    h_normalized = F.normalize(h, p=2, dim=-1)

    # 4. Perform forward pass
    output = ngpt_block(h_normalized)

    # 5. Check output shape
    assert (
        output.shape == h.shape
    ), f"Expected output shape {h.shape}, but got {output.shape}"

    # 6. Check if the output is normalized
    output_norms = torch.linalg.norm(output, dim=-1)
    assert torch.allclose(
        output_norms, torch.ones_like(output_norms)
    ), "Output should be normalized."


def test_ngpt_with_fastfeedforward_gradients() -> None:
    """Tests if gradients flow correctly through nGPT with FastFeedForward."""
    batch_size = 2
    seq_len = 8
    dim = 32
    depth = 2

    fff = FastFeedForward(
        dim=dim,
        depth=depth,
        soft_routing_during_train=True,  # Use soft routing for grad test
    )
    ngpt_block = NGPT(
        feedforward_net=fff,
        dim=dim,
    )
    ngpt_block.train()  # Set to training mode

    h = torch.randn(batch_size, seq_len, dim)
    h_normalized = F.normalize(h, p=2, dim=-1)
    h_normalized.requires_grad = True

    output = ngpt_block(h_normalized)
    loss = output.mean()
    loss.backward()

    # Check gradient for nGPT's alpha_m
    assert ngpt_block.alpha_m.grad is not None, "alpha_m should have gradients."

    # Check gradients for FastFeedForward's parameters
    # With soft routing, all routers and experts should have grads.
    for router in fff.routers:
        assert router.weight.grad is not None, "Router weights should have gradients."
        assert router.bias.grad is not None, "Router biases should have gradients."

    for expert in fff.experts:
        for param in expert.parameters():
            assert (
                param.grad is not None
            ), "All expert parameters should have gradients."
