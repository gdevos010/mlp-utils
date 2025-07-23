import torch

from einops import rearrange
from torch import nn


class SpatialGatingUnit(nn.Module):
    """Implements the Spatial Gating Unit (SGU) for the gMLP model.

    The SGU enables cross-token communication by applying a linear projection
    across the sequence dimension. This projection is learned and applied to a
    normalized version of the input tensor.

    Args:
        dim (int): The dimension of the input tensor.
        seq_len (int): The length of the input sequence.
    """

    def __init__(self, dim, seq_len) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(seq_len, seq_len, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the SpatialGatingUnit."""
        shortcut = x
        x_norm = self.norm(x)
        x_gated = rearrange(x_norm, "b n d -> b d n")
        x_gated = self.proj(x_gated)
        x_gated = rearrange(x_gated, "b d n -> b n d")
        return shortcut * x_gated


class GMLPBlock(nn.Module):
    """Implements a single gMLP block as described in the paper "Pay Attention to MLPs".

    This block applies a Gated Linear Unit with a Spatial Gating Unit (SGU) to enable
    cross-token communication. The input is first projected to a higher-dimensional space,
    split into two parts for the gate and the value, and then combined after the SGU
    operation.

    Args:
        dim (int): The input and output dimension of the block.
        dim_ff (int): The inner dimension of the feedforward network.
        seq_len (int): The length of the input sequence.
    """

    def __init__(self, dim, dim_ff, seq_len) -> None:
        super().__init__()
        self.proj_in = nn.Linear(dim, dim_ff * 2, bias=False)
        self.activation = nn.GELU()

        self.sgu = SpatialGatingUnit(dim_ff, seq_len)
        self.proj_out = nn.Linear(dim_ff, dim)

    def forward(self, x):
        u, v = self.proj_in(x).chunk(2, dim=-1)
        u = self.activation(u)
        v = self.sgu(v)
        x = u * v
        x = self.proj_out(x)
        return x


class GMLP(nn.Module):
    """Implements the gMLP model from the paper "Pay Attention to MLPs".

    This model consists of a series of gMLP blocks, which use Gated Linear Units
    with Spatial Gating Units to model relationships in sequential data without
    requiring self-attention mechanisms.

    Args:
        dim (int): The dimension of the input tokens.
        dim_ff (int): The inner dimension of the feedforward network in each block.
        seq_len (int): The length of the input sequence.
        depth (int): The number of gMLP blocks to stack.
    """

    def __init__(self, dim, dim_ff, seq_len, depth) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [GMLPBlock(dim=dim, dim_ff=dim_ff, seq_len=seq_len) for _ in range(depth)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
