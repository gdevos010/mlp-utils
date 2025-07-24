import torch

from mlp_utils.layers import FastFeedForward


def test_fastfeedforward_initialization() -> None:
    """Tests if the FastFeedForward layer can be initialized."""
    fff = FastFeedForward(
        dim=256,
        depth=3,
        mult=4,
        glu_variant="swiglu",
    )
    fff = torch.compile(fff)
    assert fff is not None, "FastFeedForward layer should be initialized."
    assert fff.depth == 3, "Depth should be set correctly."
    assert fff.num_experts == 8, "Number of experts should be 2**depth."
    assert len(fff.routers) == 7, "Number of routers should be 2**depth - 1."
    assert len(fff.experts) == 8, "Number of experts should be correct."


def test_feedforward_initialization_with_proj() -> None:
    """Tests if the FeedForward layer has the proj attribute."""
    ff = FastFeedForward(dim=256, depth=3, mult=4, glu_variant="swiglu").experts[0]
    assert hasattr(ff, "proj"), "FeedForward layer should have a proj attribute."


def test_fastfeedforward_forward_pass_expert_dim() -> None:
    """Tests the forward pass of the FastFeedForward layer with expert dim."""
    batch_size = 4
    seq_len = 16
    dim = 256
    expert_dim = 128

    fff = FastFeedForward(
        dim=dim,
        expert_dim=expert_dim,
        depth=3,
        mult=4,
        glu_variant="swiglu",
    )
    fff = torch.compile(fff)

    # Create a random input tensor
    x = torch.randn(batch_size, seq_len, dim)

    # Perform the forward pass
    output = fff(x)

    # Check if the output shape is correct
    assert output.shape == x.shape, (
        f"Expected output shape {x.shape}, but got {output.shape}"
    )


def test_fastfeedforward_forward_pass() -> None:
    """Tests the forward pass of the FastFeedForward layer."""
    batch_size = 4
    seq_len = 16
    dim = 256

    fff = FastFeedForward(
        dim=dim,
        depth=3,
        mult=4,
        glu_variant="swiglu",
    )
    fff = torch.compile(fff)
    # Create a random input tensor
    x = torch.randn(batch_size, seq_len, dim)

    # Perform the forward pass
    output = fff(x)

    # Check if the output shape is correct
    assert output.shape == x.shape, (
        f"Expected output shape {x.shape}, but got {output.shape}"
    )


def test_fastfeedforward_forward_pass_pre_norm() -> None:
    """Tests the forward pass of the FastFeedForward layer with pre-normalization."""
    batch_size = 4
    seq_len = 16
    dim = 256

    fff = FastFeedForward(
        dim=dim,
        depth=3,
        mult=4,
        glu_variant="swiglu",
        pre_norm=True,
    )
    fff = torch.compile(fff)
    # Create a random input tensor
    x = torch.randn(batch_size, seq_len, dim)

    # Perform the forward pass
    output = fff(x)

    # Check if the output shape is correct
    assert output.shape == x.shape, (
        f"Expected output shape {x.shape}, but got {output.shape}"
    )


def test_fastfeedforward_eval_mode() -> None:
    """Tests the forward pass of the FastFeedForward layer in evaluation mode."""
    batch_size = 4
    seq_len = 16
    dim = 256

    fff = FastFeedForward(
        dim=dim,
        depth=3,
        mult=4,
        glu_variant="swiglu",
    )
    fff.eval()  # Switch to evaluation mode to test hard routing
    fff = torch.compile(fff)
    x = torch.randn(batch_size, seq_len, dim)
    output = fff(x)

    assert (
        output.shape == x.shape
    ), f"Expected output shape {x.shape}, but got {output.shape}"



def test_fastfeedforward_soft_routing_grad() -> None:
    """Tests if gradients flow through the soft routing mechanism."""
    batch_size = 2
    seq_len = 8
    dim = 32

    fff = FastFeedForward(
        dim=dim,
        depth=3,
        mult=4,
        glu_variant="swiglu",
        soft_routing_during_train=True,
    )
    fff.train()  # Ensure training mode
    fff = torch.compile(fff)
    x = torch.randn(batch_size, seq_len, dim)
    output = fff(x)
    loss = output.mean()
    loss.backward()

    # Check that all routers have gradients
    for router in fff.routers:
        assert router.weight.grad is not None, "Router weights should have gradients."
        assert router.bias.grad is not None, "Router biases should have gradients."

    # Check that all experts have gradients
    for expert in fff.experts:
        for param in expert.parameters():
            assert (
                param.grad is not None
            ), "All expert parameters should have gradients in soft routing."


def test_fastfeedforward_hard_routing_grad() -> None:
    """
    Tests gradient flow in hard routing (when soft_routing_during_train=False).
    With hard routing, router parameters should not receive gradients.
    """
    dim = 32
    depth = 3

    fff = FastFeedForward(
        dim=dim,
        depth=depth,
        mult=4,
        glu_variant="swiglu",
        soft_routing_during_train=False,  # Use hard routing
    )
    fff.train()
    fff = torch.compile(fff)    
    x = torch.randn(1, 1, dim)  # Single token
    output = fff(x)
    loss = output.mean()
    loss.backward()

    # Check router gradients
    num_router_grads = 0
    for router in fff.routers:
        if router.weight.grad is not None:
            num_router_grads += 1
    assert (
        num_router_grads == 0
    ), f"Expected 0 routers to have gradients with hard routing, but got {num_router_grads}."

    # Check expert gradients
    num_expert_grads = 0
    for expert in fff.experts:
        if all(p.grad is not None for p in expert.parameters()):
            num_expert_grads += 1
    assert (
        num_expert_grads == 1
    ), f"Expected 1 expert to have gradients, but got {num_expert_grads}." 
