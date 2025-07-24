"""Custom activation functions."""

import torch
import torch.nn.functional as F

from torch import nn


class ReluSquared(nn.Module):
    """Computes the square of the ReLU of the input.

    Optionally, the output can be signed.
    """

    def __init__(self, signed: bool = False) -> None:
        super().__init__()
        self.signed = signed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for ReluSquared."""
        out = F.relu(x).square()
        if not self.signed:
            return out
        return out * torch.sign(x)


class Gelu2(nn.Module):
    """Computes the square of the GELU of the input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Gelu2."""
        return F.gelu(x).pow(2)


class BSiLU(nn.Module):
    """BSiLU activation function.

    Equation 7 from https://arxiv.org/html/2405.02207v1
    """

    def __init__(self, alpha: float = 1.67) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for BSiLU."""
        # Equation 7 from https://arxiv.org/html/2505.22074v1
        return (x + self.alpha) * torch.sigmoid(x) - self.alpha / 2


class NeLU(nn.Module):
    """NeLU activation function, often used as the backward function in a Straight-Through-Estimator."""

    def __init__(self, alpha: float = 0.05) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for NeLU."""
        return -self.alpha / (1.0 + x.square())


class Sugar(nn.Module):
    """A straight-through estimator that uses the backward_fn only for the negative part of the input."""

    def __init__(self, forward_fn: nn.Module, backward_fn: nn.Module) -> None:
        super().__init__()
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Sugar."""
        forward_out = self.forward_fn(x)

        if not x.requires_grad:
            return forward_out

        backward_out = self.backward_fn(x)

        # only neg region for backward function gradients
        soft = torch.where(x > 0, forward_out, backward_out)

        # straight-through during training
        return soft + (forward_out - soft).detach()


class StraightThroughEstimator(nn.Module):
    """A straight-through estimator."""

    def __init__(self, forward_fn: nn.Module, backward_fn: nn.Module) -> None:
        super().__init__()
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for StraightThroughEstimator."""
        hard = self.forward_fn(x)

        if not x.requires_grad:
            return hard

        soft = self.backward_fn(x)
        return soft + (hard - soft).detach()


class ReluNelu(Sugar):
    """An activation that uses ReLU in the forward pass and NeLU in the backward pass for the negative part.

    This was found to be effective in transformer models.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        """Initializes the ReluNelu activation."""
        super().__init__(nn.ReLU(), NeLU(alpha))


class SugarReLU(StraightThroughEstimator):
    """A straight-through estimator that uses ReLU in the forward pass and sigmoid in the backward pass."""

    def __init__(self) -> None:
        super().__init__(nn.ReLU(), nn.Sigmoid())
