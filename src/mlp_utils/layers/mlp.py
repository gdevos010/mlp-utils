"""Implements a standardized MLP module."""

from typing import Literal

import torch

from torch import nn

from .init_weights import initialize_weights
from .residual import ResidualWrapper


class MLP(nn.Module):
    """A standardized MLP module."""

    def __init__(  # noqa: PLR0913, C901
        self,
        input_dim: int,
        output_dim: int | None = None,
        hidden_factor: int = 4,
        dropout: float = 0.1,
        act_fn: type[nn.Module] = nn.GELU,
        use_norm: bool = True,
        norm_layer: type[nn.Module] = nn.RMSNorm,
        residual: bool = False,
        norm_mode: Literal["pre", "post", "both", "none"] = "post",
    ) -> None:
        """Initializes the MLP module.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension (defaults to input_dim if None)
            hidden_factor: Multiplier for hidden dimension (hidden_dim = input_dim * hidden_factor)
            dropout: Dropout probability
            act_fn: Activation function
            use_norm: Whether to use normalization layers
            norm_layer: Normalization layer class to use
            residual: Whether to add a residual connection
            norm_mode: Where to apply normalization relative to the first linear layer. One of
                "pre" (apply normalization to input), "post" (apply normalization after the first
                linear layer), "both" (apply both pre and post), or "none" (no normalization). Ignored
                if use_norm is False.
        """
        super().__init__()

        if hidden_factor <= 0:
            raise ValueError(f"hidden_factor must be positive, got {hidden_factor}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0,1.0), got {dropout}")

        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim

        hidden_dim = self.input_dim * hidden_factor
        layers = []

        # Determine effective normalization mode
        effective_norm_mode: Literal["pre", "post", "both", "none"] = (
            norm_mode if use_norm else "none"
        )

        # Pre-normalization if enabled
        if effective_norm_mode in ("pre", "both"):
            layers.append(norm_layer(input_dim))

        # First linear layer and normalization
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        if effective_norm_mode in ("post", "both"):
            layers.append(norm_layer(hidden_dim))

        # Handle both class and instance activations
        if isinstance(act_fn, type):
            layers.append(act_fn())
        else:
            layers.append(act_fn)

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Output layer (no implicit normalization at the output; governed by norm_mode only)
        layers.append(nn.Linear(hidden_dim, self.output_dim))

        # Create the model
        self.model = nn.Sequential(*layers)

        for name, module in self.model.named_modules():
            initialize_weights(module, layer_name=name)

        # If using residual connection and input_dim == output_dim
        if residual and self.input_dim == self.output_dim:
            self.model = ResidualWrapper(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLP.

        Args:
            x: Tensor of shape (..., input_dim)

        Returns:
            Tensor of shape (..., output_dim)
        """
        return self.model(x)
