import pytest
import torch

from torch import nn

from mlp_utils.layers.gating import GatingMechanism


@pytest.fixture
def input_tensor():
    return torch.randn(10, 20)


def test_gating_mechanism_bottleneck_factor(input_tensor):
    gate = GatingMechanism(input_dim=20, bottleneck_factor=0.5)
    output = gate(input_tensor)
    assert output.shape == (10, 20)
    assert output.min() >= 0
    assert output.max() <= 1


def test_gating_mechanism_bottleneck_dim(input_tensor):
    gate = GatingMechanism(input_dim=20, bottleneck_dim=15)
    output = gate(input_tensor)
    assert output.shape == (10, 20)


def test_gating_mechanism_output_dim(input_tensor):
    gate = GatingMechanism(input_dim=20, bottleneck_dim=10, output_dim=5)
    output = gate(input_tensor)
    assert output.shape == (10, 5)


def test_gating_mechanism_pre_norm(input_tensor):
    gate = GatingMechanism(input_dim=20, bottleneck_dim=10, pre_norm=True)
    # Check if the first layer is a norm layer
    assert isinstance(gate.gate[0], nn.modules.normalization.RMSNorm)
    output = gate(input_tensor)
    assert output.shape == (10, 20)


def test_gating_mechanism_no_norm(input_tensor):
    gate = GatingMechanism(input_dim=20, bottleneck_dim=10, use_norm=False)
    # Check that no norm layers are present
    assert not any(isinstance(m, nn.modules.normalization.RMSNorm) for m in gate.gate)
    output = gate(input_tensor)
    assert output.shape == (10, 20)


def test_gating_mechanism_no_sigmoid(input_tensor):
    gate = GatingMechanism(input_dim=20, bottleneck_dim=10, sigmoid_output=False)
    output = gate(input_tensor)
    # Without sigmoid, output is not guaranteed to be in [0, 1]
    assert output.min() < 0 or output.max() > 1
    assert output.shape == (10, 20)


def test_gating_mechanism_assertions():
    with pytest.raises(AssertionError):
        # Both bottleneck_factor and bottleneck_dim are None
        GatingMechanism(input_dim=20)
    with pytest.raises(AssertionError):
        # Both bottleneck_factor and bottleneck_dim are provided
        GatingMechanism(input_dim=20, bottleneck_factor=0.5, bottleneck_dim=10)


def test_gating_mechanism_custom_activation(input_tensor):
    gate = GatingMechanism(input_dim=20, bottleneck_dim=10, act_fn=nn.ReLU)
    output = gate(input_tensor)
    assert output.shape == (10, 20)


def test_gating_mechanism_dropout(input_tensor):
    gate = GatingMechanism(input_dim=20, bottleneck_dim=10, dropout=0.5)
    gate.train()  # Set to train mode to activate dropout
    output = gate(input_tensor)
    # In train mode with dropout, output should be different from input
    # and some values should be zeroed out (scaled by 1/p)
    assert output.shape == (10, 20)

    gate.eval()  # Set to eval mode to deactivate dropout
    output_eval = gate(input_tensor)
    assert output_eval.shape == (10, 20)
