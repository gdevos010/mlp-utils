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
        """Forward pass with residual connection.

        Args:
            x: Input tensor

        Returns:
            Output with residual connection
        """
        return x + self.module(x)
