import pytest
import torch

from mlp_utils.layers.mlp import MLP


def test_mlp_initialization() -> None:
    """Tests basic initialization of the MLP."""
    mlp = MLP(input_dim=64, output_dim=32, hidden_factor=2)
    assert isinstance(mlp, torch.nn.Module)


def test_mlp_forward_pass() -> None:
    """Tests the forward pass of the MLP."""
    input_tensor = torch.randn(16, 64)
    mlp = MLP(input_dim=64, output_dim=32, hidden_factor=2)
    output_tensor = mlp(input_tensor)
    assert output_tensor.shape == (16, 32)


def test_mlp_output_dim_defaults_to_input_dim() -> None:
    """Tests that output_dim defaults to input_dim."""
    mlp = MLP(input_dim=64)
    assert mlp.output_dim == 64
    input_tensor = torch.randn(16, 64)
    output_tensor = mlp(input_tensor)
    assert output_tensor.shape == (16, 64)


def test_mlp_residual_connection() -> None:
    """Tests the residual connection."""
    input_tensor = torch.randn(16, 64)
    mlp = MLP(input_dim=64, output_dim=64, residual=True)
    output_tensor = mlp(input_tensor)
    assert output_tensor.shape == input_tensor.shape


def test_mlp_residual_connection_no_wrapper_on_dim_mismatch() -> None:
    """Tests that ResidualWrapper is not applied if input_dim != output_dim."""
    from mlp_utils.layers.residual import ResidualWrapper

    mlp_no_residual = MLP(input_dim=64, output_dim=32, residual=True)
    assert not isinstance(mlp_no_residual.model, ResidualWrapper)


def test_mlp_pre_norm() -> None:
    """Tests the pre-normalization functionality."""
    mlp = MLP(input_dim=64, pre_norm=True, use_norm=True)
    first_layer = list(mlp.model.children())[0]
    from torch.nn import RMSNorm

    assert isinstance(first_layer, RMSNorm)


def test_mlp_invalid_hidden_factor() -> None:
    """Tests for ValueError with invalid hidden_factor."""
    with pytest.raises(ValueError, match="hidden_factor must be positive, got 0"):
        MLP(input_dim=64, hidden_factor=0)
    with pytest.raises(ValueError, match="hidden_factor must be positive, got -1"):
        MLP(input_dim=64, hidden_factor=-1)


def test_mlp_invalid_dropout() -> None:
    """Tests for ValueError with invalid dropout."""
    with pytest.raises(ValueError, match="dropout must be in \\[0.0,1.0\\), got -0.1"):
        MLP(input_dim=64, dropout=-0.1)
    with pytest.raises(ValueError, match="dropout must be in \\[0.0,1.0\\), got 1.0"):
        MLP(input_dim=64, dropout=1.0)


def test_dropout_layer_presence() -> None:
    """Tests that dropout layer is present when specified."""
    mlp_with_dropout = MLP(input_dim=64, dropout=0.5)
    has_dropout = any(
        isinstance(m, torch.nn.Dropout) for m in mlp_with_dropout.model.modules()
    )
    assert has_dropout

    mlp_no_dropout = MLP(input_dim=64, dropout=0.0)
    has_dropout_false = any(
        isinstance(m, torch.nn.Dropout) for m in mlp_no_dropout.model.modules()
    )
    assert not has_dropout_false
