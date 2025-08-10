"""Implements Gated Linear Units (GLU) and Masked Gated Linear Units (MGLU)."""

import torch

from torch import nn


class _GLUBase(nn.Module):
    """Base class for Gated Linear Units (GLU).

    Implements y = value ⊙ activation(gate), where `[value, gate] = Linear(x)`
    after projecting to size `2 * dim_out` along the last dimension.

    Args:
      dim_in (int): Input feature dimension.
      dim_out (int): Output feature dimension.
      activation (nn.Module): Activation applied to the gate branch.
      bias (bool, optional): If True, includes a bias term in the projection.
        Defaults to True.

    Shapes:
      - x: (..., dim_in)
      - y: (..., dim_out)
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
        """Apply the GLU transformation.

        Args:
            x (torch.Tensor): Input of shape (..., dim_in) and floating dtype.

        Returns:
            torch.Tensor: Output of shape (..., dim_out) with the same dtype as `x`.
        """
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.activation(gate)


class GLU(_GLUBase):
    """Gated Linear Unit with a sigmoid activation.

    Shapes:
      - x: (..., dim_in)
      - y: (..., dim_out)
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.Sigmoid(), bias=bias)


class Bilinear(_GLUBase):
    """GLU with identity activation (no nonlinearity).

    Computes an elementwise product of the two linear branches: value ⊙ gate.

    Shapes:
      - x: (..., dim_in)
      - y: (..., dim_out)
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.Identity(), bias=bias)


class ReGLU(_GLUBase):
    """GLU with ReLU activation on the gate.

    Shapes:
      - x: (..., dim_in)
      - y: (..., dim_out)
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.ReLU(), bias=bias)


class SwiGLU(_GLUBase):
    """GLU with Swish (SiLU) activation on the gate.

    Shapes:
      - x: (..., dim_in)
      - y: (..., dim_out)
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.SiLU(), bias=bias)


class GeGLU(_GLUBase):
    """GLU with GELU activation on the gate.

    Shapes:
      - x: (..., dim_in)
      - y: (..., dim_out)
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.GELU(), bias=bias)


class _MGLUBase(nn.Module):
    """Base class for Masked Gated Linear Units (MGLU).

    Mixes between a linear path and a gated nonlinearity per output channel
    using a learnable mask trained with a straight-through estimator (STE).

    Let v = Linear(x). A sigmoid surrogate is applied to the mask parameter to
    produce a near-binary mixing coefficient m_hat ∈ [0, 1]. The output is:

      y = m_hat ⊙ (v ⊙ activation(v)) + (1 - m_hat) ⊙ v.

    Args:
      dim_in (int): Input feature dimension.
      dim_out (int): Output feature dimension.
      activation (nn.Module): Activation used on the gated path.
      bias (bool, optional): If True, includes a bias term in the projection.
        Defaults to True.

    Attributes:
      proj (nn.Linear): Linear projection from dim_in to dim_out.
      activation (nn.Module): Activation module applied on the gated path.
      mask (torch.nn.Parameter): Learnable mask of shape (dim_out,).

    Shapes:
      - x: (..., dim_in)
      - y: (..., dim_out)
      - mask: (dim_out,) broadcast to (1, ..., dim_out)

    References:
      - Tajima et al., "Masked Gated Linear Unit" (`https://arxiv.org/abs/2506.23225`).
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
        """Apply the MGLU transformation.

        Args:
            x (torch.Tensor): Input of shape (..., dim_in) and floating dtype.

        Returns:
            torch.Tensor: Output of shape (..., dim_out) with the same dtype as `x`.

        Notes:
            The learnable mask parameter has shape `(dim_out,)` and is broadcast
            over the leading dimensions. A sigmoid surrogate is used to construct
            a straight-through estimator (STE) for gradients.
        """
        projected = self.proj(x)

        # Binarize the mask using a straight-through estimator (STE).
        # Use a continuous surrogate (sigmoid) to preserve gradients.
        temperature = 1.0  # STE sharpness; consider exposing as a hyperparameter
        mask_surrogate = torch.sigmoid(self.mask / temperature)
        binary_mask_hard = (self.mask >= 0).to(projected.dtype)
        binary_mask = (binary_mask_hard - mask_surrogate).detach() + mask_surrogate

        # Ensure mask is broadcastable to projected's shape.
        mask_view = (1,) * (projected.dim() - 1) + (-1,)
        binary_mask = binary_mask.view(mask_view)

        # The core GLU operation is value * activation(gate).
        # In MGLU, the projected tensor serves as both value and gate.
        # The mask determines where the gating is applied.
        gated_out = projected * self.activation(projected)

        # Combine the gated and non-gated parts using a differentiable mixture.
        # Avoid boolean branching so gradients flow to the mask via the surrogate.
        return binary_mask * gated_out + (1.0 - binary_mask) * projected


class MGLU(_MGLUBase):
    """MGLU with a sigmoid activation on the gated path.

    Shapes:
      - x: (..., dim_in)
      - y: (..., dim_out)
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.Sigmoid(), bias=bias)


class BilinearMGLU(_MGLUBase):
    """MGLU with identity activation (no nonlinearity) on the gated path.

    On masked channels, the gated path computes v ⊙ v (elementwise square).
    The per-channel STE-learned mask mixes between v ⊙ v and the linear
    pass-through v.

    Shapes:
      - x: (..., dim_in)
      - y: (..., dim_out)
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.Identity(), bias=bias)


class ReMGLU(_MGLUBase):
    """MGLU with ReLU activation on the gated path.

    Shapes:
      - x: (..., dim_in)
      - y: (..., dim_out)
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.ReLU(), bias=bias)


class SwiMGLU(_MGLUBase):
    """MGLU with Swish (SiLU) activation on the gated path.

    Shapes:
      - x: (..., dim_in)
      - y: (..., dim_out)
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.SiLU(), bias=bias)


class GeMGLU(_MGLUBase):
    """MGLU with GELU activation on the gated path.

    Shapes:
      - x: (..., dim_in)
      - y: (..., dim_out)
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__(dim_in, dim_out, nn.GELU(), bias=bias)
