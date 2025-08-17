# ABOUTME: Differentiable Tversky similarity utilities and projection layer stubs.
# ABOUTME: Provides function and module scaffolding to be implemented in later steps.
"""Tversky similarity functions and a projection layer (scaffold).

This module provides the public API for Tversky-based similarity and a
`TverskyProjection` layer that projects inputs onto a bank of prototypes using the
Tversky similarity. Implementations are added in subsequent steps.

Reference: Tversky Neural Networks: Psychologically Plausible Deep Learning with
Differentiable Tversky Similarity (arXiv:2506.11035).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from torch import nn

__all__ = [
    "tversky_similarity",
    "pairwise_tversky",
    "TverskyProjection",
]


def tversky_similarity(  # noqa: PLR0913, PLR0912
    input: torch.Tensor,  # noqa: A002
    prototype: torch.Tensor,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    eps: float = 1e-6,
    input_transform: str | Callable[[torch.Tensor], torch.Tensor] | None = None,
    nonnegative: bool = True,
    smoothing_tau: float | None = None,
) -> torch.Tensor:
    """Compute the (differentiable) Tversky similarity between two tensors.

    Uses per-feature proxy set operations to estimate intersection and distinct
    parts, then applies the Tversky similarity:

        S = (I + eps) / (I + alpha * A_only + beta * B_only + eps)

    where I = sum(min(x, y)), A_only = sum(relu(x - y)), B_only = sum(relu(y - x)).
    Optional smoothing via `smoothing_tau` replaces hard min and ReLU with smooth
    approximations that approach the hard ops as tau -> 0.

    Args:
        input: Input tensor of shape (..., D).
        prototype: Prototype tensor of shape (..., D), broadcastable with `input`.
        alpha: Weight for input-only mass.
        beta: Weight for prototype-only mass.
        eps: Numerical stability constant added to numerator/denominator.
        input_transform: Optional transform applied before set-op proxies. One of
            None, "relu", "clamp01", "sigmoid", or a callable. If provided, it
            supersedes `nonnegative`.
        nonnegative: If True and no explicit transform is provided, clamp to
            nonnegative via ReLU.
        smoothing_tau: Optional temperature for smoothed proxies; if provided,
            must be > 0.

    Returns:
        Tensor containing similarity scores with shape equal to the broadcasted
        leading dims of `input`/`prototype` without the last feature dimension.

    Raises:
        ValueError: If `smoothing_tau` is provided and <= 0, or if an unknown
            `input_transform` string is supplied.
    """
    # Validate smoothing temperature
    if smoothing_tau is not None and smoothing_tau <= 0:
        raise ValueError(
            f"smoothing_tau must be > 0 when provided, got {smoothing_tau}"
        )

    # Apply optional input transform
    if input_transform is not None:
        if isinstance(input_transform, str):
            if input_transform == "relu":
                x = F.relu(input)
                y = F.relu(prototype)
            elif input_transform == "clamp01":
                x = torch.clamp(input, 0.0, 1.0)
                y = torch.clamp(prototype, 0.0, 1.0)
            elif input_transform == "sigmoid":
                x = torch.sigmoid(input)
                y = torch.sigmoid(prototype)
            else:
                raise ValueError(
                    f"Unknown input_transform: {input_transform!r}. "
                    "Expected one of 'relu', 'clamp01', 'sigmoid', or a callable."
                )
        elif callable(input_transform):
            x = input_transform(input)
            y = input_transform(prototype)
        else:
            raise ValueError(
                "input_transform must be None, a recognized string, or a callable"
            )
    elif nonnegative:
        x = F.relu(input)
        y = F.relu(prototype)
    else:
        x = input
        y = prototype

    # Hard vs smoothed set-operation proxies
    if smoothing_tau is None:
        intersection = torch.minimum(x, y).sum(dim=-1)
        a_only = F.relu(x - y).sum(dim=-1)
        b_only = F.relu(y - x).sum(dim=-1)
    else:
        tau = float(smoothing_tau)
        # Smooth minimum via soft-min: -tau * logsumexp([-x/tau, -y/tau])
        stacked = torch.stack((x, y), dim=-1)  # (..., D, 2)
        soft_min = -tau * torch.logsumexp(-stacked / tau, dim=-1)  # (..., D)
        intersection = soft_min.sum(dim=-1)

        # Smooth ReLU via tau * softplus((z)/tau)
        a_only = (tau * F.softplus((x - y) / tau)).sum(dim=-1)
        b_only = (tau * F.softplus((y - x) / tau)).sum(dim=-1)

    similarity = (intersection + eps) / (
        intersection + alpha * a_only + beta * b_only + eps
    )
    return similarity


def pairwise_tversky(  # noqa: PLR0913
    input: torch.Tensor,  # noqa: A002
    prototypes: torch.Tensor,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    eps: float = 1e-6,
    input_transform: str | Callable[[torch.Tensor], torch.Tensor] | None = None,
    nonnegative: bool = True,
    smoothing_tau: float | None = None,
) -> torch.Tensor:
    """Compute pairwise Tversky similarities between inputs and a bank of prototypes.

    This is a scaffold stub. The implementation will vectorize over prototypes
    and support broadcasting over leading dimensions of `input`.

    Args:
        input: Tensor of shape ([...,] D).
        prototypes: Tensor of shape (K, D).
        alpha: Weight for input-only mass.
        beta: Weight for prototype-only mass.
        eps: Numerical stability constant.
        input_transform: Optional transform applied before proxy set operations.
        nonnegative: If True and no explicit transform is provided, clamp to nonnegative.
        smoothing_tau: Optional temperature for smoothed proxies; if provided,
            must be > 0.

    Returns:
        Tensor of shape ([...,] K) with similarity scores.

    Raises:
        NotImplementedError: This function is a scaffold and not yet implemented.
    """
    raise NotImplementedError


class TverskyProjection(nn.Module):
    """Layer that projects inputs via Tversky similarity against learned prototypes.

    This is a scaffold stub. The final implementation will register learnable
    prototype weights of shape [output_dim, input_dim], optional bias, and
    optional learnable alpha/beta parameters, and will compute pairwise Tversky
    similarities in the forward pass.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        output_dim: int,
        *,
        alpha: float = 0.5,
        beta: float = 0.5,
        eps: float = 1e-6,
        bias: bool = False,
        input_transform: str | Callable[[torch.Tensor], torch.Tensor] | None = None,
        nonnegative: bool = True,
        smoothing_tau: float | None = None,
        learnable_alpha: bool = False,
        learnable_beta: bool = False,
        alpha_beta_normalize: bool = False,
        temperature: float | None = None,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward pass (scaffold).

        Raises:
            NotImplementedError: This method is a scaffold and not yet implemented.
        """
        raise NotImplementedError
