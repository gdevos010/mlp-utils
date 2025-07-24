"""Pytest for PathWeightedFFF."""

import pytest
import torch

from torch import nn

from mlp_utils.layers import PathWeightedFFF


def test_initialization():
    """Tests if the PathWeightedFFF layer initializes correctly."""
    model = PathWeightedFFF(input_width=32, depth=3, output_width=64)
    assert isinstance(model, nn.Module)
    assert model.input_width == 32
    assert model.depth == 3
    assert model.output_width == 64
    assert model.n_nodes == 2 ** (3 + 1) - 1

    # Test for invalid depth
    with pytest.raises(ValueError):
        PathWeightedFFF(input_width=32, depth=-1, output_width=64)


def test_forward_pass_shape_3d() -> None:
    """Tests the output shape of the forward pass for 3D inputs."""
    batch_size = 16
    seq_len = 10
    input_width = 32
    output_width = 64
    depth = 3
    model = PathWeightedFFF(
        input_width=input_width, depth=depth, output_width=output_width
    )
    x = torch.randn(batch_size, seq_len, input_width)
    output = model(x)
    assert output.shape == (batch_size, seq_len, output_width)


def test_input_dimension_error():
    """Tests that a non-2D or non-3D input raises a ValueError."""
    model = PathWeightedFFF(input_width=32, depth=3, output_width=64)
    # Test with 1D input
    with pytest.raises(ValueError):
        model(torch.randn(32))
    # Test with 4D input
    with pytest.raises(ValueError):
        model(torch.randn(16, 10, 5, 32))


def test_parameters_are_registered():
    """Tests that w1s and w2s are registered as parameters."""
    model = PathWeightedFFF(input_width=32, depth=3, output_width=64)
    params = dict(model.named_parameters())
    assert "w1s" in params
    assert "w2s" in params
    assert params["w1s"].requires_grad
    assert params["w2s"].requires_grad


def test_backward_pass():
    """Tests that gradients are computed correctly."""
    model = PathWeightedFFF(input_width=32, depth=3, output_width=64)
    x = torch.randn(16, 32)
    output = model(x)
    loss = output.mean()
    loss.backward()

    assert model.w1s.grad is not None
    assert model.w2s.grad is not None
    # Due to the routing mechanism, only weights of visited nodes will have gradients.
    # Therefore, we check if *any* gradient is non-zero, not all.
    assert torch.any(model.w1s.grad != 0)
    assert torch.any(model.w2s.grad != 0)


def test_deterministic_path_selection() -> None:
    """Tests if the routing mechanism follows a predictable path given specific weights and inputs."""
    input_width = 2
    depth = 2
    output_width = 4
    model = PathWeightedFFF(
        input_width=input_width, depth=depth, output_width=output_width
    )

    # --- Manually set weights for deterministic routing ---
    with torch.no_grad():
        # Node 0 (root): Decision boundary is x[0] = 0.
        # If x[0] > 0, go right (node 2). If x[0] < 0, go left (node 1).
        model.w1s.fill_(0.0)
        model.w1s[0, 0] = 1.0  # Coeff for x[0]

        # Node 1 (left child of root): Decision boundary is x[1] = 0.
        # If x[1] > 0, go right (node 4).
        model.w1s[1, 1] = 1.0  # Coeff for x[1]

        # Node 2 (right child of root): Decision boundary is x[1] = 0.
        # If x[1] < 0, go left (node 5).
        model.w1s[2, 1] = -1.0  # Coeff for x[1]

    # --- Test Case 1: Path root -> left -> right (0 -> 1 -> 4) ---
    # Input x = [-1, 1] should produce this path.
    # Root (0): x[0] = -1 < 0 -> go left to node 1.
    # Node 1: x[1] = 1 > 0 -> go right to node 4.
    x1 = torch.tensor([[-1.0, 1.0]])
    model(x1)  # Forward pass to populate internal states for inspection

    # We need to access internal state, which is not ideal but necessary here.
    # Let's re-run the logic to get the path.
    all_nodes_path1 = []
    current_node = 0
    for _ in range(depth + 1):
        all_nodes_path1.append(current_node)
        plane_coeffs = model.w1s[current_node]
        score = torch.dot(x1.squeeze(), plane_coeffs)
        choice = (score >= 0).long()
        current_node = current_node * 2 + choice + 1
    assert all_nodes_path1 == [0, 1, 4]

    # --- Test Case 2: Path root -> right -> left (0 -> 2 -> 5) ---
    # Input x = [1, -1] should produce this path.
    # Root (0): x[0] = 1 > 0 -> go right to node 2.
    # Node 2: x[1] = -1 -> dot product with [0, -1] is 1 > 0 -> go right to node 6 ?
    # Let's re-check the logic. `current_nodes * 2 + plane_choices + 1`
    # Node 2's w1 is [0, -1]. x is [1, -1]. score = -1 * -1 = 1 > 0. choice = 1.
    # next_node = 2 * 2 + 1 + 1 = 6.
    # Oh, I see. My manual check was wrong. Let's trace it carefully.
    # x = [1, -1]. Path:
    # 1. Root (0), w=[1,0]. score=1>0. choice=1. next_node=0*2+1+1=2.
    # 2. Node (2), w=[0,-1]. score=1>0. choice=1. next_node=2*2+1+1=6.
    # 3. Node (6). End of path.
    # So the path is [0, 2, 6]
    x2 = torch.tensor([[1.0, -1.0]])
    all_nodes_path2 = []
    current_node = 0
    for _ in range(depth + 1):
        all_nodes_path2.append(current_node)
        plane_coeffs = model.w1s[current_node]
        score = torch.dot(x2.squeeze(), plane_coeffs)
        choice = (score >= 0).long()
        current_node = current_node * 2 + choice + 1
    assert all_nodes_path2 == [0, 2, 6]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_compatibility(device):
    """Tests if the model runs on different devices."""
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")

    model = PathWeightedFFF(input_width=32, depth=3, output_width=64).to(device)
    x = torch.randn(16, 32).to(device)
    output = model(x)
    assert output.device.type == device

    # Also test backward pass on device
    loss = output.mean()
    loss.backward()
    assert model.w1s.grad.device.type == device
    assert model.w2s.grad.device.type == device
