import pytest
import torch

from mlp_utils.layers.feedforward import FeedForward

BATCH_SIZE = 4
DIM_IN = 16
MULT = 2

ALL_GLU_VARIANTS = [
    "none",
    "glu",
    "geglu",
    "swiglu",
    "reglu",
    "bilinear",
    "mglu",
    "mgeglu",
    "mswiglu",
    "mreglu",
    "mbilinear",
]


@pytest.fixture
def input_tensor() -> torch.Tensor:
    """Returns a random tensor for testing."""
    return torch.randn(BATCH_SIZE, DIM_IN)


def test_feedforward_initialization() -> None:
    """Tests basic initialization of the FeedForward module."""
    ff = FeedForward(dim=DIM_IN)
    assert isinstance(ff, torch.nn.Module)


@pytest.mark.parametrize("glu_variant", ALL_GLU_VARIANTS)
def test_feedforward_forward_pass(input_tensor: torch.Tensor, glu_variant: str) -> None:
    """Tests the forward pass for all GLU variants."""
    ff = FeedForward(dim=DIM_IN, mult=MULT, glu_variant=glu_variant)
    output_tensor = ff(input_tensor)
    assert output_tensor.shape == (BATCH_SIZE, DIM_IN)


def test_feedforward_pre_norm(input_tensor: torch.Tensor) -> None:
    """Tests the pre-normalization functionality."""
    ff = FeedForward(dim=DIM_IN, pre_norm=True)
    assert ff.pre_norm is not None
    # Check that it runs
    ff(input_tensor)


def test_feedforward_invalid_glu_variant() -> None:
    """Tests that an invalid GLU variant raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown GLU variant: invalid_variant"):
        FeedForward(dim=DIM_IN, glu_variant="invalid_variant")


def test_dropout_presence_and_value() -> None:
    """Tests that dropout layer is present and has the correct value."""
    ff_with_dropout = FeedForward(dim=DIM_IN, dropout=0.5)
    dropout_layers = [
        m for m in ff_with_dropout.net.modules() if isinstance(m, torch.nn.Dropout)
    ]
    assert len(dropout_layers) > 0
    assert dropout_layers[0].p == 0.5

    ff_no_dropout = FeedForward(dim=DIM_IN, dropout=0.0)
    dropout_layers_zero = [
        m for m in ff_no_dropout.net.modules() if isinstance(m, torch.nn.Dropout)
    ]
    assert len(dropout_layers_zero) > 0
    assert dropout_layers_zero[0].p == 0.0


def test_feedforward_custom_activation() -> None:
    """Tests using a custom activation function."""
    ff = FeedForward(dim=DIM_IN, glu_variant="none", activation=torch.nn.ReLU)
    assert isinstance(ff.net[1], torch.nn.ReLU)


def test_feedforward_custom_norm_layer(input_tensor: torch.Tensor) -> None:
    """Tests using a custom normalization layer."""
    ff = FeedForward(dim=DIM_IN, pre_norm=True, norm_layer=torch.nn.LayerNorm)
    assert isinstance(ff.pre_norm, torch.nn.LayerNorm)
    # Check that it runs
    ff(input_tensor)


def test_proj_attribute_no_glu() -> None:
    """Tests the proj attribute when no GLU variant is used."""
    ff = FeedForward(dim=DIM_IN, mult=MULT, glu_variant="none")
    assert ff.proj is ff.net[0]
    assert isinstance(ff.proj, torch.nn.Linear)
    assert ff.proj.in_features == DIM_IN
    assert ff.proj.out_features == DIM_IN * MULT


@pytest.mark.parametrize("glu_variant", list(set(ALL_GLU_VARIANTS) - {"none"}))
def test_proj_attribute_with_glu(glu_variant: str) -> None:
    """Tests the proj attribute when a GLU variant is used."""
    ff = FeedForward(dim=DIM_IN, mult=MULT, glu_variant=glu_variant)
    assert ff.proj is ff.net[0].proj
    assert isinstance(ff.proj, torch.nn.Linear)
    assert ff.proj.in_features == DIM_IN

    hidden_dim = DIM_IN * MULT
    # Most GLU variants project to double the hidden dimension for the gate and value
    if "m" in glu_variant:  # Masked GLU
        expected_out_features = hidden_dim
    else:
        expected_out_features = 2 * hidden_dim

    assert ff.proj.out_features == expected_out_features
