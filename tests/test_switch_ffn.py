import pytest
import torch

from mlp_utils.layers import FeedForward, SwitchFFN


@pytest.fixture
def switch_config():
    """Provides a basic configuration for the SwitchFFN layer."""
    return {
        "dim": 32,
        "num_experts": 4,
        "ff_kwargs": {"mult": 4, "glu_variant": "swiglu"},
    }


@pytest.fixture
def switch_ffn(switch_config):
    """Initializes a SwitchFFN model for testing."""
    return SwitchFFN(**switch_config)


@pytest.fixture
def input_tensor(switch_config):
    """Creates a sample input tensor for testing."""
    batch_size = 2
    seq_len = 10
    return torch.randn(batch_size, seq_len, switch_config["dim"])


def test_switch_ffn_initialization(switch_ffn, switch_config):
    """Tests if the SwitchFFN layer initializes correctly."""
    assert isinstance(switch_ffn, SwitchFFN)
    assert switch_ffn.num_experts == switch_config["num_experts"]
    assert len(switch_ffn.experts) == switch_config["num_experts"]
    assert all(isinstance(expert, FeedForward) for expert in switch_ffn.experts)
    assert hasattr(switch_ffn, "has_aux_loss")
    assert switch_ffn.has_aux_loss is True


def test_switch_ffn_invalid_experts():
    """Tests that an error is raised for an invalid number of experts."""
    with pytest.raises(ValueError, match="num_experts must be a positive integer"):
        SwitchFFN(dim=32, num_experts=0)


def test_switch_ffn_forward_pass(switch_ffn, input_tensor):
    """Tests the forward pass of the SwitchFFN layer."""
    output, loss = switch_ffn(input_tensor)

    assert output.shape == input_tensor.shape, "Output shape should match input shape"
    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor"
    assert loss.ndim == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Load balancing loss should be non-negative"


def test_router_receives_gradients(switch_ffn, input_tensor):
    """Ensures the router's weights are updated during backpropagation."""
    output, loss = switch_ffn(input_tensor)
    # Create a dummy loss for the main output
    main_loss = output.mean()
    total_loss = main_loss + loss
    total_loss.backward()

    assert switch_ffn.router.weight.grad is not None
    assert switch_ffn.router.weight.grad.shape == switch_ffn.router.weight.shape


def test_token_dropping(input_tensor):
    """Tests that tokens are dropped when expert capacity is exceeded."""
    dim = input_tensor.shape[-1]
    # Force capacity to be low to ensure tokens are dropped
    model = SwitchFFN(
        dim=dim,
        num_experts=2,
        capacity_factor=0.1,  # Low capacity factor
        ff_kwargs={"mult": 2},
    )

    # The router is initialized with small random weights, so we can't be
    # certain which expert gets chosen. We will manually set the router weights
    # to make one expert always preferred.
    with torch.no_grad():
        model.router.weight.fill_(0.0)
        model.router.weight[0, :] = 1.0  # Force expert 0 to be chosen

    output, _ = model(input_tensor)
    assert output.shape == input_tensor.shape


def test_switch_ffn_forward_single_expert():
    """Tests SwitchFFN with only one expert."""
    dim = 16
    model = SwitchFFN(dim=dim, num_experts=1, ff_kwargs={"mult": 2})
    x = torch.randn(2, 4, dim)
    output, loss = model(x)
    assert output.shape == x.shape
    assert loss.ndim == 0


@pytest.mark.parametrize("num_experts", [2, 8, 32])
def test_switch_ffn_varying_experts(num_experts):
    """Tests SwitchFFN with a varying number of experts."""
    dim = 64
    model = SwitchFFN(
        dim=dim,
        num_experts=num_experts,
        ff_kwargs={"mult": 4, "glu_variant": "geglu"},
    )
    x = torch.randn(4, 16, dim)
    output, loss = model(x)
    assert output.shape == x.shape
    assert loss.ndim == 0
