import pytest
import torch

from mlp_utils.layers.gmlp import GMLP, GMLPBlock, SpatialGatingUnit


@pytest.fixture
def sample_input():
    """
    Provides a sample input tensor for testing.
    """
    return torch.randn(4, 16, 128)  # (batch_size, seq_len, dim)


def test_spatial_gating_unit_output_shape(sample_input):
    """
    Tests if the SpatialGatingUnit produces an output with the correct shape.
    """
    batch_size, seq_len, dim = sample_input.shape
    sgu = SpatialGatingUnit(dim=dim, seq_len=seq_len)
    output = sgu(sample_input)
    assert output.shape == sample_input.shape, f"Expected shape {sample_input.shape}, but got {output.shape}"


def test_gmlp_block_output_shape(sample_input):
    """
    Tests if the GMLPBlock produces an output with the correct shape.
    """
    batch_size, seq_len, dim = sample_input.shape
    dim_ff = 256
    gmlp_block = GMLPBlock(dim=dim, dim_ff=dim_ff, seq_len=seq_len)
    output = gmlp_block(sample_input)
    assert output.shape == (batch_size, seq_len, dim), f"Expected shape {(batch_size, seq_len, dim)}, but got {output.shape}"


def test_gmlp_output_shape(sample_input):
    """
    Tests if the GMLP model produces an output with the correct shape.
    """
    batch_size, seq_len, dim = sample_input.shape
    dim_ff = 256
    depth = 6
    gmlp = GMLP(dim=dim, dim_ff=dim_ff, seq_len=seq_len, depth=depth)
    output = gmlp(sample_input)
    assert output.shape == (batch_size, seq_len, dim), f"Expected shape {(batch_size, seq_len, dim)}, but got {output.shape}"


def test_gmlp_forward_pass(sample_input):
    """
    Tests that the GMLP forward pass runs without errors.
    """
    dim = sample_input.shape[-1]
    seq_len = sample_input.shape[1]
    gmlp = GMLP(dim=dim, dim_ff=256, seq_len=seq_len, depth=2)
    try:
        output = gmlp(sample_input)
        assert output is not None
    except Exception as e:
        pytest.fail(f"gMLP forward pass failed with exception: {e}") 
