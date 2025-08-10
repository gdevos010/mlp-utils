"""A PyTorch implementation of gMLP."""

import torch

from einops import rearrange
from torch import nn


class SpatialGatingUnit(nn.Module):
    """Spatial Gating Unit (SGU) for gMLP.

    Performs a learned linear projection across the sequence dimension on a
    normalized representation, enabling cross-token communication.

    Args:
        dim (int): Feature dimension of each token.
        seq_len (int): Sequence length (projection dimension).
    """

    def __init__(self, dim: int, seq_len: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(seq_len, seq_len)
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Spatial Gating Unit.

        Args:
            x (torch.Tensor): Input of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output of shape (batch_size, seq_len, dim).
        """
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
        dim (int): Input and output model dimension.
        dim_ff (int): Feedforward inner (hidden) dimension.
        seq_len (int): Sequence length (for SGU).
    """

    def __init__(self, dim: int, dim_ff: int, seq_len: int) -> None:
        super().__init__()
        self.proj_in = nn.Linear(dim, dim_ff * 2)
        self.activation = nn.GELU()

        self.sgu = SpatialGatingUnit(dim_ff, seq_len)
        self.proj_out = nn.Linear(dim_ff, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gMLP block.

        Args:
            x (torch.Tensor): Input of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output of shape (batch_size, seq_len, dim).
        """
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
        dim (int): Token feature dimension.
        dim_ff (int): Inner (hidden) dimension for each block.
        seq_len (int): Sequence length.
        depth (int): Number of blocks.
    """

    def __init__(self, dim: int, dim_ff: int, seq_len: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    GMLPBlock(dim=dim, dim_ff=dim_ff, seq_len=seq_len),
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gMLP stack.

        Args:
            x (torch.Tensor): Input of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output of shape (batch_size, seq_len, dim).
        """
        for block in self.blocks:
            x = block(x) + x
        return x
