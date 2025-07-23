from typing import Literal

import torch

from torch import nn

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
        Activation used for the **value** branch in `'geglu'` mode and inside the
        vanilla MLP path. Ignored for `'swiglu'`.
    glu_variant : Literal["none","glu","geglu","swiglu"], default "none"
        • **"none"** - conventional two-layer MLP
        • **"glu"** - classic GLU: _value·σ(gate)_
        • **"geglu"** - GeGLU: _GELU(value)·gate_
        • **"swiglu"** - SwiGLU: _SiLU(value)·gate_
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
        glu_variant: Literal["none", "glu", "geglu", "swiglu"] = "none",
        pre_norm: bool = False,
        norm_layer: type[nn.Module] = nn.RMSNorm,
    ) -> None:
        super().__init__()

        hidden_dim: int = int(dim * mult)
        self.glu_variant = glu_variant.lower()
        self.pre_norm = norm_layer(dim) if pre_norm else None
        self.dropout = nn.Dropout(dropout)

        self.net = nn.Sequential()
        if self.glu_variant != "none":
            self.proj_in = nn.Linear(dim, hidden_dim * 2)
            self.proj_out = nn.Linear(hidden_dim, dim)

            if self.glu_variant == "glu":  # value · σ(gate)
                self.value_act: nn.Module = nn.Identity()
                self.gate_act: nn.Module = nn.Sigmoid()

            elif self.glu_variant == "geglu":  # GELU(value) · gate
                self.value_act = activation()
                self.gate_act = nn.Identity()

            elif self.glu_variant == "swiglu":  # SiLU(value) · gate
                self.value_act = nn.SiLU()
                self.gate_act = nn.Identity()

            else:
                raise ValueError(f"Unknown GLU variant: {self.glu_variant}")

            initialize_weights(self.proj_in, init_method="default")
            initialize_weights(self.proj_out, init_method="default", scale=0.1)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        if self.glu_variant != "none":
            value, gate = self.proj_in(x).chunk(2, dim=-1)
            x = self.value_act(value) * self.gate_act(gate)
            x = self.dropout(x)
            return self.proj_out(x)

        return self.net(x)
