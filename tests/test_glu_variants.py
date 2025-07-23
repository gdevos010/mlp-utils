import pytest
import torch

from torch import nn

from mlp_utils.layers.glu import (
    GLU,
    MGLU,
    Bilinear,
    BilinearMGLU,
    GeGLU,
    GeMGLU,
    ReGLU,
    ReMGLU,
    SwiGLU,
    SwiMGLU,
)


@pytest.mark.parametrize(
    ("glu_class", "activation"),
    [
        (GLU, nn.Sigmoid),
        (Bilinear, nn.Identity),
        (ReGLU, nn.ReLU),
        (SwiGLU, nn.SiLU),
        (GeGLU, nn.GELU),
    ],
)
def test_glu_variants(glu_class, activation) -> None:
    dim_in = 16
    dim_out = 32
    batch_size = 4

    glu = glu_class(dim_in, dim_out)
    x = torch.randn(batch_size, dim_in)
    output = glu(x)

    assert output.shape == (batch_size, dim_out)

    # Check activation function
    assert isinstance(glu.activation, activation)

    # test with no bias
    glu_no_bias = glu_class(dim_in, dim_out, bias=False)
    output_no_bias = glu_no_bias(x)
    assert output_no_bias.shape == (batch_size, dim_out)


@pytest.mark.parametrize(
    ("mglu_class", "activation"),
    [
        (MGLU, nn.Sigmoid),
        (BilinearMGLU, nn.Identity),
        (ReMGLU, nn.ReLU),
        (SwiMGLU, nn.SiLU),
        (GeMGLU, nn.GELU),
    ],
)
def test_mglu_variants(mglu_class, activation) -> None:
    dim_in = 16
    dim_out = 32
    batch_size = 4

    mglu = mglu_class(dim_in, dim_out)
    x = torch.randn(batch_size, dim_in)
    output = mglu(x)

    assert output.shape == (batch_size, dim_out)

    # Check activation function
    assert isinstance(mglu.activation, activation)

    # Check mask properties
    assert isinstance(mglu.mask, nn.Parameter)
    assert mglu.mask.shape == (dim_out,)

    # test with no bias
    mglu_no_bias = mglu_class(dim_in, dim_out, bias=False)
    output_no_bias = mglu_no_bias(x)
    assert output_no_bias.shape == (batch_size, dim_out)
