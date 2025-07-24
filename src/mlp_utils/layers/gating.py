"""Implements a standardized gating mechanism."""

import torch

from torch import nn

from .init_weights import initialize_weights


class GatingMechanism(nn.Module):
    """A standardized gating mechanism."""

    def __init__(  # noqa: PLR0913, C901
        self,
        input_dim: int,
        output_dim: int | None = None,
        bottleneck_factor: float | None = None,
        bottleneck_dim: int | None = None,
        dropout: float = 0.1,
        act_fn: type[nn.Module] = nn.GELU,
        use_norm: bool = True,
        norm_layer: type[nn.Module] = nn.RMSNorm,
        sigmoid_output: bool = True,
        pre_norm: bool = False,
    ) -> None:
        """Create a standardized gating mechanism.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            bottleneck_factor: Factor to reduce dimension in bottleneck (bottleneck_dim = input_dim * bottleneck_factor). Recommended value is 0.5
            bottleneck_dim: Explicit bottleneck dimension, overrides bottleneck_factor if provided
            dropout: Dropout probability
            act_fn: Activation function
            use_norm: Whether to use normalization layers
            norm_layer: Normalization layer class to use
            sigmoid_output: Whether to apply sigmoid activation to the output
            pre_norm: Whether to apply normalization before or after the first linear layer
        """
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        assert not (bottleneck_factor is None and bottleneck_dim is None), (
            "Either bottleneck_factor or bottleneck_dim must be provided"
        )
        assert not (bottleneck_factor is not None and bottleneck_dim is not None), (
            "Both bottleneck_factor and bottleneck_dim cannot be set simultaneously"
        )

        if bottleneck_dim is None:
            assert bottleneck_factor is not None
            bottleneck_dim = int(input_dim * bottleneck_factor)

        layers = []

        # Pre-normalization if enabled
        if pre_norm and use_norm:
            layers.append(norm_layer(input_dim))

        # First linear layer
        layers.append(nn.Linear(input_dim, bottleneck_dim))

        # Normalization after first layer
        if use_norm and not pre_norm:
            layers.append(norm_layer(bottleneck_dim))

        # Handle both class and instance activations
        if isinstance(act_fn, type):
            layers.append(act_fn())
        else:
            layers.append(act_fn)

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(bottleneck_dim, output_dim))

        # Output activation
        if sigmoid_output:
            layers.append(nn.Sigmoid())

        self.gate = nn.Sequential(*layers)
        for name, module in self.gate.named_modules():
            initialize_weights(module, init_method="gating", layer_name=name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the gating mechanism."""
        return self.gate(x)
