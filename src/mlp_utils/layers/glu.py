"""Implements Gated Linear Units (GLU) and Masked Gated Linear Units (MGLU)."""

import torch

from torch import nn


class _GLUBase(nn.Module):
    """Base class for Gated Linear Units.

    This class implements the core logic of a GLU, which consists of a projection
    followed by a gated activation. The specific activation function is provided
    by the subclasses.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: nn.Module,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the GLU.

        Args:
            x (torch.Tensor): The input tensor of shape (..., dim_in).

        Returns:
            torch.Tensor: The output tensor of shape (..., dim_out).
        """
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.activation(gate)


class GLU(_GLUBase):
    """Gated Linear Unit with a sigmoid activation."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.Sigmoid(), bias=bias)


class Bilinear(_GLUBase):
    """Gated Linear Unit with no activation.

    This results in a bilinear-like interaction (x * Wx), where one part
    of the projection gates the other without a non-linearity.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.Identity(), bias=bias)


class ReGLU(_GLUBase):
    """Gated Linear Unit with ReLU activation."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.ReLU(), bias=bias)


class SwiGLU(_GLUBase):
    """Gated Linear Unit with Swish (SiLU) activation."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.SiLU(), bias=bias)


class GeGLU(_GLUBase):
    """Gated Linear Unit with GELU activation."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.GELU(), bias=bias)


class _MGLUBase(nn.Module):
    """Base class for Masked Gated Linear Units.

    This class implements the core logic of a MGLU, which uses a single
    shared weight matrix for the gate and value projections, with a learnable
    mask to differentiate between them. The mask is binarized during the
    forward pass using a straight-through estimator.

    This implementation is a naive PyTorch version based on the paper
    "Masked Gated Linear Unit" by Tajima et al. (https://arxiv.org/abs/2506.23225).
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: nn.Module,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.activation = activation
        # Learnable mask parameter, initialized randomly.
        # It will be binarized in the forward pass.
        self.mask = nn.Parameter(torch.randn(dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MGLU.

        Args:
            x (torch.Tensor): The input tensor of shape (..., dim_in).

        Returns:
            torch.Tensor: The output tensor of shape (..., dim_out).
        """
        projected = self.proj(x)

        # Binarize the mask using a straight-through estimator.
        # The hard binarization is used in the forward pass, but gradients
        # are allowed to flow back to the original mask parameter.
        binary_mask_hard = torch.ge(self.mask, 0).to(projected.dtype)
        binary_mask = (binary_mask_hard - self.mask).detach() + self.mask

        # Ensure mask is broadcastable to projected's shape.
        mask_view = (1,) * (projected.dim() - 1) + (-1,)
        binary_mask = binary_mask.view(mask_view)

        # The core GLU operation is value * activation(gate).
        # In MGLU, the projected tensor serves as both value and gate.
        # The mask determines where the gating is applied.
        gated_out = projected * self.activation(projected)

        # Combine the gated and non-gated parts using the mask.
        return torch.where(binary_mask.to(torch.bool), gated_out, projected)


class MGLU(_MGLUBase):
    """Masked Gated Linear Unit with a sigmoid activation."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.Sigmoid(), bias=bias)


class BilinearMGLU(_MGLUBase):
    """Masked Gated Linear Unit with no activation."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.Identity(), bias=bias)


class ReMGLU(_MGLUBase):
    """Masked Gated Linear Unit with ReLU activation."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.ReLU(), bias=bias)


class SwiMGLU(_MGLUBase):
    """Masked Gated Linear Unit with Swish (SiLU) activation."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.SiLU(), bias=bias)


class GeMGLU(_MGLUBase):
    """Masked Gated Linear Unit with GELU activation."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.GELU(), bias=bias)
