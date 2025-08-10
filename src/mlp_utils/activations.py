"""Custom activation functions."""

import torch
import torch.nn.functional as F

from torch import nn


class ReluSquared(nn.Module):
    """Compute the square of ReLU(x).

    Optionally multiplies by `sign(x)` to produce a signed output.
    """

    def __init__(self, signed: bool = False) -> None:
        super().__init__()
        self.signed = signed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the ReluSquared activation.

        Args:
            x (torch.Tensor): Input tensor of any shape and floating dtype.

        Returns:
            torch.Tensor: Output tensor with the same shape and dtype as `x`.
        """
        out = F.relu(x).square()
        if not self.signed:
            return out
        return out * torch.sign(x)


class Gelu2(nn.Module):
    """Compute the square of GELU(x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Gelu2 activation.

        Args:
            x (torch.Tensor): Input tensor of any shape and floating dtype.

        Returns:
            torch.Tensor: Output tensor with the same shape and dtype as `x`.
        """
        return F.gelu(x).pow(2)


class BSiLU(nn.Module):
    """BSiLU activation function.

    Implements Equation 7 from the referenced paper.
    See: https://arxiv.org/html/2505.22074v1
    """

    def __init__(self, alpha: float = 1.67) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the BSiLU activation.

        Args:
            x (torch.Tensor): Input tensor of any shape and floating dtype.

        Returns:
            torch.Tensor: Output tensor with the same shape and dtype as `x`.
        """
        # Equation 7 from https://arxiv.org/html/2505.22074v1
        return (x + self.alpha) * torch.sigmoid(x) - self.alpha / 2


class NeLU(nn.Module):
    """NeLU activation function.

    Often used as the backward function in a straight-through estimator.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the NeLU activation.

        Args:
            x (torch.Tensor): Input tensor of any shape and floating dtype.

        Returns:
            torch.Tensor: Output tensor with the same shape and dtype as `x`.
        """
        return -self.alpha / (1.0 + x.square())


class Sugar(nn.Module):
    """Straight-through estimator with custom negative-region gradients.

    Uses the provided `forward_fn` in the forward pass. In the backward pass,
    gradients for the negative region use `backward_fn`.
    """

    def __init__(self, forward_fn: nn.Module, backward_fn: nn.Module) -> None:
        super().__init__()
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Sugar estimator.

        Args:
            x (torch.Tensor): Input tensor of any shape and floating dtype.

        Returns:
            torch.Tensor: Output tensor with the same shape and dtype as `x`.
        """
        forward_out = self.forward_fn(x)

        if not x.requires_grad:
            return forward_out

        backward_out = self.backward_fn(x)

        # only neg region for backward function gradients
        soft = torch.where(x > 0, forward_out, backward_out)

        # straight-through during training
        return soft + (forward_out - soft).detach()


class StraightThroughEstimator(nn.Module):
    """Straight-through estimator wrapper.

    Uses a hard function in the forward pass and a soft surrogate in backward.
    """

    def __init__(self, forward_fn: nn.Module, backward_fn: nn.Module) -> None:
        super().__init__()
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the straight-through estimator.

        Args:
            x (torch.Tensor): Input tensor of any shape and floating dtype.

        Returns:
            torch.Tensor: Output tensor with the same shape and dtype as `x`.
        """
        hard = self.forward_fn(x)

        if not x.requires_grad:
            return hard

        soft = self.backward_fn(x)
        return soft + (hard - soft).detach()


class ReluNelu(Sugar):
    """ReLU forward with NeLU gradients on the negative region.

    Often effective in transformer models.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        """Initialize the ReluNelu activation.

        Args:
            alpha (float): NeLU alpha parameter.
        """
        super().__init__(nn.ReLU(), NeLU(alpha))


class SugarReLU(StraightThroughEstimator):
    """Straight-through estimator with ReLU forward and sigmoid backward."""

    def __init__(self) -> None:
        super().__init__(nn.ReLU(), nn.Sigmoid())
