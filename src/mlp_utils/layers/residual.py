"""Implements a residual wrapper for any module."""

import torch

from torch import nn


class ResidualWrapper(nn.Module):
    """Wrapper to add residual connection to any module."""

    def __init__(self, module: nn.Module) -> None:
        """Initialize residual wrapper.

        Args:
            module: Module to wrap with residual connection
        """
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the wrapped module with a residual connection.

        Args:
            x (torch.Tensor): Input tensor of any shape where the last dimension
                matches the wrapped module's expected input.

        Returns:
            torch.Tensor: Output tensor with the same shape as `x`.
        """
        return x + self.module(x)
