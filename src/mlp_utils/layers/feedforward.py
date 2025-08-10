"""Implements a feed-forward block with optional gated variants."""

from typing import Literal

import torch

from torch import nn

from .glu import (
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
from .init_weights import initialize_weights


class FeedForward(nn.Module):
    """Feed-forward block with optional GLU variants.

    This module projects the last dimension, so it supports inputs of rank 2+
    as long as the trailing dimension equals `dim`.

    Args:
        dim (int): Input and output feature dimension.
        mult (int): Expansion factor for the hidden layer. Defaults to 4.
        dropout (float): Dropout probability applied after the gate or activation.
            Defaults to 0.0.
        activation (type[nn.Module]): Activation for the vanilla MLP path (ignored
            for GLU variants). Defaults to `nn.GELU`.
        glu_variant (Literal["none","glu","geglu","swiglu","reglu","bilinear",
            "mglu","mgeglu","mswiglu","mreglu","mbilinear"]): Selects the GLU
            variant or a conventional MLP when "none". Defaults to "none".
        pre_norm (bool): If True, applies `norm_layer` before projections.
            Defaults to False.
        norm_layer (type[nn.Module]): Normalization layer class. Defaults to `nn.RMSNorm`.
    """

    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
        glu_variant: Literal[
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
        ] = "none",
        pre_norm: bool = False,
        norm_layer: type[nn.Module] = nn.RMSNorm,
    ) -> None:
        super().__init__()

        hidden_dim: int = int(dim * mult)
        self.glu_variant = glu_variant.lower()
        self.pre_norm = norm_layer(dim) if pre_norm else None

        if self.glu_variant != "none":
            glu_map = {
                "glu": GLU,
                "geglu": GeGLU,
                "swiglu": SwiGLU,
                "reglu": ReGLU,
                "bilinear": Bilinear,
                "mglu": MGLU,
                "mgeglu": GeMGLU,
                "mswiglu": SwiMGLU,
                "mreglu": ReMGLU,
                "mbilinear": BilinearMGLU,
            }
            glu_class = glu_map.get(self.glu_variant)
            if glu_class is None:
                raise ValueError(f"Unknown GLU variant: {self.glu_variant}")

            glu_layer = glu_class(dim, hidden_dim)
            output_proj = nn.Linear(hidden_dim, dim)

            initialize_weights(glu_layer.proj, init_method="default")
            initialize_weights(output_proj, init_method="default", scale=0.1)

            self.net = nn.Sequential(
                glu_layer,
                nn.Dropout(dropout),
                output_proj,
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                activation(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
            )

            # Initialize all Linear layers, scaling down the final projection.
            # Robustly detect the last Linear layer instead of relying on module names.
            linear_layers = [m for m in self.net.modules() if isinstance(m, nn.Linear)]
            if linear_layers:
                for layer in linear_layers[:-1]:
                    initialize_weights(layer, init_method="default")
                initialize_weights(linear_layers[-1], init_method="default", scale=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward block.

        Args:
            x (torch.Tensor): Input of shape (..., dim) and floating dtype.

        Returns:
            torch.Tensor: Output of shape (..., dim) with the same dtype as `x`.
        """
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        return self.net(x)
