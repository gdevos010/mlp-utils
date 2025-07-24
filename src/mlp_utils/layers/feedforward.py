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
    r"""Feed-forward block with optional gated variants.

    Parameters
    ----------
    dim : int
        Input / output dimension.
    mult : int, default 4
        Expansion factor for the hidden layer.
    dropout : float, default 0.0
        Dropout probability applied *after* the gate.
    activation : nn.Module, default nn.GELU
        Activation used for the vanilla MLP path. Ignored for GLU variants.
    glu_variant : Literal[
        "none", "glu", "geglu", "swiglu", "reglu", "bilinear",
        "mglu", "mgeglu", "mswiglu", "mreglu", "mbilinear"
        ], default "none"
        - **"none"** - conventional two-layer MLP
        - **"glu"** - classic GLU: _value·σ(gate)_
        - **"geglu"** - GeGLU: _value·GELU(gate)_
        - **"swiglu"** - SwiGLU: _value·SiLU(gate)_
        - **"reglu"** - ReGLU: _value·ReLU(gate)_
        - **"bilinear"** - Bilinear: _value·gate_
        - **"mglu"** - Masked GLU with Sigmoid
        - **"mgeglu"** - Masked GLU with GELU
        - **"mswiglu"** - Masked GLU with SiLU
        - **"mreglu"** - Masked GLU with ReLU
        - **"mbilinear"** - Masked Bilinear (Identity activation)
    pre_norm : bool, default False
        If ``True`` a normalization layer (``norm_layer``) is applied *before*
        the projections.
    norm_layer : nn.Module, default nn.RMSNorm
        Normalization layer class.
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
            self.proj = glu_layer.proj
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                activation(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
            )

            for name, module in self.net.named_modules():
                if isinstance(module, nn.Linear):
                    if name == "3":  # Last layer in the sequential
                        initialize_weights(module, init_method="default", scale=0.1)
                    else:
                        initialize_weights(module, init_method="default")
            self.proj = self.net[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        return self.net(x)
