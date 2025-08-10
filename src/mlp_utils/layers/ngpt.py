"""Implements the MLP block of the Normalized Transformer (nGPT) as described in "nGPT: Normalized Transformer with Representation Learning on the Hypersphere".

Inspiration from https://github.com/lucidrains/nGPT-pytorch/
(https://arxiv.org/html/2410.01131v2)
"""

from typing import TypeVar

import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from .feedforward import FeedForward

T = TypeVar("T")


def exists(v: T | None) -> bool:
    """Check if a value is not None."""
    return v is not None


def default(v: T | None, d: T) -> T:
    """Return v if it exists, otherwise return d."""
    return v if exists(v) else d


def l2norm(t: torch.Tensor, dim: int = -1, norm_eps: float = 0.0) -> torch.Tensor:
    """L2-normalize a tensor with optional soft normalization.

    Args:
        t (torch.Tensor): Input tensor of any shape and floating dtype.
        dim (int): Dimension along which to compute the norm. Defaults to -1.
        norm_eps (float): Soft normalization epsilon. If `0.0`, uses strict unit
            normalization. Otherwise, scales by a detached target norm in
            `[1 - norm_eps, 1 + norm_eps]`. Defaults to 0.0.

    Returns:
        torch.Tensor: Tensor of the same shape and dtype as `t`, normalized along `dim`.
    """
    if norm_eps == 0.0:
        return F.normalize(t, p=2, dim=dim)
    else:
        eps = 1e-5 if t.dtype == torch.float16 else 1e-10
        norm = t.norm(dim=dim, keepdim=True)
        target_norm = norm.detach().clamp(min=1.0 - norm_eps, max=1.0 + norm_eps)
        divisor = norm / target_norm
        return t / divisor.clamp(min=eps)


class L2Norm(nn.Module):
    """Module wrapper for `l2norm`, usable with parametrizations."""

    def __init__(self, dim: int = -1, norm_eps: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.norm_eps = norm_eps

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Apply L2 normalization.

        Args:
            t (torch.Tensor): Input tensor of any shape and floating dtype.

        Returns:
            torch.Tensor: Tensor with the same shape and dtype as `t`, L2-normalized.
        """
        return l2norm(t, dim=self.dim, norm_eps=self.norm_eps)


class NormLinear(nn.Module):
    """A linear layer with normalized weights."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        norm_dim_in: bool = True,
        norm_eps: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=False)
        norm_dim = -1 if norm_dim_in else 0
        register_parametrization(
            self.linear, "weight", L2Norm(dim=norm_dim, norm_eps=norm_eps)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the normalized linear transformation.

        Args:
            x (torch.Tensor): Input of shape (..., dim_in).

        Returns:
            torch.Tensor: Output of shape (..., dim_out).
        """
        return self.linear(x)


class Scale(nn.Module):
    """A learnable scaling factor.

    As described in Section 2.5 of the nGPT paper.
    """

    def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), scale))
        self.forward_scale = init / scale

    def forward(self) -> torch.Tensor:
        """Return the current scale vector.

        Returns:
            torch.Tensor: A 1D tensor of shape (dim,) representing the scaling factors.
        """
        return self.scale * self.forward_scale


class NGPT(nn.Module):
    """nGPT MLP block with hyperspherical representation updates.

    Wraps a feedforward network with the nGPT update rule. Both input and output
    hidden states are L2-normalized along the last dimension.
    See: "nGPT: Normalized Transformer with Representation Learning on the Hypersphere"
    (`https://arxiv.org/html/2410.01131v2`).
    """

    def __init__(
        self,
        dim: int,
        feedforward_net: nn.Module | None = None,
        scalar_alpha: bool = False,
        alpha_m_init: float = 0.5,
        alpha_m_scale: float | None = None,
        norm_eps: float = 0.0,
    ) -> None:
        """Initializes the nGPT MLP block.

        Args:
            dim (int): The feature dimension of the hidden states.
            feedforward_net (nn.Module, optional): The feedforward network (MLP) to be wrapped.
                                                   If None, a `FeedForward` network with SwiGLU is used.
                                                   Defaults to None.
            scalar_alpha (bool): If True, use a single scalar for alpha_m.
                                 If False, use a vector of size `dim`. Defaults to False.
            alpha_m_init (float): Initial value for the learnable parameter alpha_m.
            alpha_m_scale (float, optional): The scale for the alpha_m parameter.
                                             Defaults to `dim ** -0.5` for vector alpha.
            norm_eps (float): Epsilon for soft normalization. Defaults to 0.0 (hard normalization).
        """
        super().__init__()
        ff_network = FeedForward(dim=dim, mult=4, glu_variant="swiglu")
        self.feedforward_net = default(feedforward_net, ff_network)
        self.norm_eps = norm_eps

        if scalar_alpha:
            alpha_dim = 1
            scale = default(alpha_m_scale, 1.0)
        else:
            alpha_dim = dim
            scale = default(alpha_m_scale, dim**-0.5)
        self.alpha_m = Scale(alpha_dim, alpha_m_init, scale)

    @property
    def needs_normalized_target(self) -> bool:
        """Specifies that the target for this model should be normalized for loss calculation."""
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the nGPT MLP block transformation.

        Args:
            x (torch.Tensor): Input of shape (..., dim). Internally L2-normalized
                along the last dimension to lie on the hypersphere.

        Returns:
            torch.Tensor: Output of shape (..., dim), L2-normalized along the last dimension.
        """
        # The nGPT block operates on normalized vectors on the hypersphere.
        # As per the paper, the input x is brought to the hypersphere.
        x_norm = l2norm(x, norm_eps=self.norm_eps)

        # x_M <- Norm(MLP(x_norm))
        x_m = self.feedforward_net(x_norm)
        x_m_norm = l2norm(x_m, norm_eps=self.norm_eps)

        # x <- Norm(x_norm + α_M * (x_M - x_norm))
        # Linear interpolation (LERP) step on the hypersphere.
        x_updated = x_norm + self.alpha_m() * (x_m_norm - x_norm)
        x_out = l2norm(x_updated, norm_eps=self.norm_eps)

        return x_out
