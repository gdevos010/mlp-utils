import pytest
import torch

from torch import nn

from mlp_utils.activations import (
    BSiLU,
    Gelu2,
    NeLU,
    ReluNelu,
    ReluSquared,
    StraightThroughEstimator,
    Sugar,
    SugarReLU,
)
from mlp_utils.layers import (
    GLU,
    GMLP,
    MGLU,
    MLP,
    NGPT,
    Bilinear,
    BilinearMGLU,
    FastFeedForward,
    FeedForward,
    GatingMechanism,
    GeGLU,
    GeMGLU,
    ReGLU,
    ReMGLU,
    ResidualWrapper,
    SwiGLU,
    SwiMGLU,
)

# A simple check to see if torch.compile is available
torch_compile_available = hasattr(torch, "compile")


# Define a very simple model for testing the residual wrapper
class SimpleModule(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.skipif(
    not torch_compile_available, reason="torch.compile is not available"
)
class TestCompiledModules:
    """Test suite for torch.compile compatibility of all modules."""

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

    @pytest.mark.parametrize(
        "activation_class",
        [ReluSquared, Gelu2, BSiLU, NeLU, SugarReLU, ReluNelu],
    )
    def test_activations_compile(self, activation_class) -> None:
        """Verify that basic activation functions can be compiled."""
        model = activation_class()
        compiled_model = torch.compile(model)
        input_tensor = torch.randn(10, 20)
        output = compiled_model(input_tensor)
        assert output is not None
        assert output.shape == input_tensor.shape

    def test_straight_through_estimator_compile(self) -> None:
        """Verify that the StraightThroughEstimator can be compiled."""
        model = StraightThroughEstimator(nn.ReLU(), nn.Sigmoid())
        compiled_model = torch.compile(model)
        input_tensor = torch.randn(10, 20)
        output = compiled_model(input_tensor)
        assert output is not None
        assert output.shape == input_tensor.shape

    def test_sugar_compile(self) -> None:
        """Verify that the Sugar STE variant can be compiled."""
        model = Sugar(nn.ReLU(), NeLU())
        compiled_model = torch.compile(model)
        input_tensor = torch.randn(10, 20)
        output = compiled_model(input_tensor)
        assert output is not None
        assert output.shape == input_tensor.shape

    @pytest.mark.parametrize(
        "glu_class",
        [GLU, Bilinear, ReGLU, SwiGLU, GeGLU],
    )
    def test_glu_variants_compile(self, glu_class) -> None:
        """Verify that all GLU variants can be compiled."""
        dim_in, dim_out = 16, 32
        model = glu_class(dim_in, dim_out)
        compiled_model = torch.compile(model)
        input_tensor = torch.randn(4, dim_in)
        output = compiled_model(input_tensor)
        assert output is not None
        assert output.shape == (4, dim_out)

    @pytest.mark.parametrize(
        "mglu_class",
        [MGLU, BilinearMGLU, ReMGLU, SwiMGLU, GeMGLU],
    )
    def test_mglu_variants_compile(self, mglu_class) -> None:
        """Verify that all MGLU variants can be compiled."""
        dim_in, dim_out = 16, 32
        model = mglu_class(dim_in, dim_out)
        compiled_model = torch.compile(model)
        input_tensor = torch.randn(4, dim_in)
        output = compiled_model(input_tensor)
        assert output is not None
        assert output.shape == (4, dim_out)

    @pytest.mark.parametrize("glu_variant", ALL_GLU_VARIANTS)
    def test_feedforward_compile(self, glu_variant) -> None:
        """Verify that FeedForward with all GLU variants can be compiled."""
        dim = 16
        model = FeedForward(dim=dim, mult=2, glu_variant=glu_variant)
        compiled_model = torch.compile(model)
        input_tensor = torch.randn(4, dim)
        output = compiled_model(input_tensor)
        assert output is not None
        assert output.shape == input_tensor.shape

    def test_fastfeedforward_compile(self) -> None:
        """Verify that FastFeedForward can be compiled in both training and eval modes."""
        dim, depth = 32, 3
        # The hard routing (eval) path is currently optimized specifically for SwiGLU.
        model = FastFeedForward(
            dim=dim, depth=depth, soft_routing_during_train=True, glu_variant="swiglu"
        )
        compiled_model = torch.compile(model)
        input_tensor = torch.randn(4, 16, dim)

        # Test soft routing (train mode)
        model.train()
        output_train = compiled_model(input_tensor)
        assert output_train is not None
        assert output_train.shape == input_tensor.shape

        # Test hard routing (eval mode)
        model.eval()
        output_eval = compiled_model(input_tensor)
        assert output_eval is not None
        assert output_eval.shape == input_tensor.shape

    def test_gating_mechanism_compile(self) -> None:
        """Verify that GatingMechanism can be compiled."""
        dim = 20
        model = GatingMechanism(input_dim=dim, bottleneck_factor=0.5)
        compiled_model = torch.compile(model)
        input_tensor = torch.randn(10, dim)
        output = compiled_model(input_tensor)
        assert output is not None
        assert output.shape == input_tensor.shape

    def test_gmlp_compile(self) -> None:
        """Verify that GMLP can be compiled."""
        dim, seq_len, depth = 32, 16, 2
        model = GMLP(dim=dim, dim_ff=dim * 2, seq_len=seq_len, depth=depth)
        compiled_model = torch.compile(model)
        input_tensor = torch.randn(4, seq_len, dim)
        output = compiled_model(input_tensor)
        assert output is not None
        assert output.shape == input_tensor.shape

    def test_mlp_compile(self) -> None:
        """Verify that MLP can be compiled."""
        dim = 64
        model = MLP(input_dim=dim, output_dim=dim, hidden_factor=2)
        compiled_model = torch.compile(model)
        input_tensor = torch.randn(16, dim)
        output = compiled_model(input_tensor)
        assert output is not None
        assert output.shape == input_tensor.shape

    def test_ngpt_compile(self) -> None:
        """Verify that NGPT can be compiled."""
        dim, depth = 32, 2
        fff = FastFeedForward(dim=dim, depth=depth)
        model = NGPT(feedforward_net=fff, dim=dim)
        compiled_model = torch.compile(model)
        input_tensor = torch.randn(4, 16, dim)
        input_normalized = torch.nn.functional.normalize(input_tensor, p=2, dim=-1)
        output = compiled_model(input_normalized)
        assert output is not None
        assert output.shape == input_tensor.shape

    def test_residual_wrapper_compile(self) -> None:
        """Verify that ResidualWrapper can be compiled."""
        dim = 16
        simple_module = SimpleModule(dim)
        model = ResidualWrapper(simple_module)
        compiled_model = torch.compile(model)
        input_tensor = torch.randn(4, dim)
        output = compiled_model(input_tensor)
        assert output is not None
        assert output.shape == input_tensor.shape
