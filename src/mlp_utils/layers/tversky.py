# ABOUTME: Differentiable Tversky similarity utilities and projection layer stubs.
# ABOUTME: Provides function and module scaffolding to be implemented in later steps.
"""Tversky similarity functions and a projection layer.

This module provides Tversky-based similarity utilities and
`TverskyProjection`, a layer that projects inputs onto a bank of prototypes
using the Tversky similarity.

Reference: Tversky Neural Networks: Psychologically Plausible Deep Learning with
Differentiable Tversky Similarity (arXiv:2506.11035).
"""

from __future__ import annotations

import math

from collections.abc import Callable

import torch
import torch.nn.functional as F

from torch import nn

__all__ = [
    "tversky_similarity",
    "pairwise_tversky",
    "TverskyProjection",
    "tversky_attributions",
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
    """Compute pairwise Tversky similarities between inputs and prototypes.

    Vectorizes over the prototype axis and supports broadcasting over all leading
    dimensions of ``input``. Internally delegates to :func:`tversky_similarity`.

    Args:
        input: Tensor of shape ``[..., D]``.
        prototypes: Tensor of shape ``[K, D]`` where each row is a prototype.
        alpha: Weight for input-only mass.
        beta: Weight for prototype-only mass.
        eps: Numerical stability constant.
        input_transform: Optional transform applied before proxy set operations.
        nonnegative: If True and no explicit transform is provided, clamp to nonnegative.
        smoothing_tau: Optional temperature for smoothed proxies; if provided, must be > 0.

    Returns:
        Tensor of shape ``[..., K]`` with similarity scores.
    """
    # Shape to enable broadcasting across prototypes: [..., 1, D] vs [K, D] -> [..., K, D]
    input_expanded = input.unsqueeze(-2)
    prototypes_expanded = prototypes.unsqueeze(0)
    sims = tversky_similarity(
        input_expanded,
        prototypes_expanded,
        alpha=alpha,
        beta=beta,
        eps=eps,
        input_transform=input_transform,
        nonnegative=nonnegative,
        smoothing_tau=smoothing_tau,
    )
    return sims


def tversky_attributions(
    input: torch.Tensor,  # noqa: A002
    prototype: torch.Tensor,
    *,
    input_transform: str | Callable[[torch.Tensor], torch.Tensor] | None = None,
    nonnegative: bool = True,
    smoothing_tau: float | None = None,
    # alpha/beta are not used to compute components themselves; they are included for API symmetry
    alpha: float = 0.5,
    beta: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-feature contributions for intersection and distinctive parts.

    Returns three tensors of the same shape as ``input`` (broadcasted against
    ``prototype``) corresponding to elementwise contributions whose sums across
    the last dimension yield the aggregate terms used by :func:`tversky_similarity`.

    Args:
        input: Tensor of shape ``[..., D]``.
        prototype: Tensor of shape broadcastable to ``[..., D]``.
        input_transform: Optional transform (``None``, ``"relu"``, ``"clamp01``,
            ``"sigmoid"`` or a callable).
        nonnegative: If True and no explicit transform is provided, clamp to nonnegative.
        smoothing_tau: If provided (>0), use smooth proxies matching
            :func:`tversky_similarity`.
        alpha: Unused parameter; included for API symmetry.
        beta: Unused parameter; included for API symmetry.

    Returns:
        Tuple ``(i_components, a_components, b_components)`` each of shape ``[..., D]``.
    """
    if smoothing_tau is not None and smoothing_tau <= 0:
        raise ValueError(
            f"smoothing_tau must be > 0 when provided, got {smoothing_tau}"
        )

    # Apply same preprocessing as similarity, minimizing branches for lint friendliness
    if input_transform is None:
        if nonnegative:
            x = F.relu(input)
            y = F.relu(prototype)
        else:
            x = input
            y = prototype
    elif isinstance(input_transform, str):
        transforms: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            "relu": F.relu,
            "clamp01": lambda t: torch.clamp(t, 0.0, 1.0),
            "sigmoid": torch.sigmoid,
        }
        try:
            transform_fn = transforms[input_transform]
        except KeyError as exc:  # noqa: PERF203 (clarity)
            raise ValueError(
                f"Unknown input_transform: {input_transform!r}. "
                "Expected one of 'relu', 'clamp01', 'sigmoid', or a callable."
            ) from exc
        x = transform_fn(input)
        y = transform_fn(prototype)
    elif callable(input_transform):
        x = input_transform(input)
        y = input_transform(prototype)
    else:
        raise ValueError(
            "input_transform must be None, a recognized string, or a callable"
        )

    if smoothing_tau is None:
        i_components = torch.minimum(x, y)
        a_components = F.relu(x - y)
        b_components = F.relu(y - x)
    else:
        tau = float(smoothing_tau)
        # Ensure broadcasted shapes align before stacking
        x_b, y_b = torch.broadcast_tensors(x, y)
        stacked = torch.stack((x_b, y_b), dim=-1)  # (..., D, 2)
        soft_min = -tau * torch.logsumexp(-stacked / tau, dim=-1)
        # Shift so that for nonnegative inputs, components remain nonnegative
        i_components = soft_min + tau * math.log(2.0)
        a_components = tau * F.softplus((x_b - y_b) / tau)
        b_components = tau * F.softplus((y_b - x_b) / tau)

    return i_components, a_components, b_components


class TverskyProjection(nn.Module):
    """Projection layer using Tversky similarity against learned prototypes.

    Parameters:
        input_dim: Feature dimension of inputs.
        output_dim: Number of prototypes (output channels).
        alpha, beta: Weights for distinctive parts in Tversky denominator. Stored as buffers.
        eps: Numerical stability constant for similarity formula.
        bias: If True, add a learnable bias of shape ``[output_dim]``.
        input_transform: Optional transform applied before proxy set operations.
        nonnegative: If True and no explicit transform is provided, clamp to nonnegative.
        smoothing_tau: Optional temperature for smoothed proxies; if provided, must be > 0.
        learnable_alpha, learnable_beta: Reserved for future extension (not enabled yet).
        alpha_beta_normalize: Reserved for future extension to renormalize alpha/beta.
        temperature: Optional post-similarity scaling factor.

    Notes:
        - With ``bias=False`` and default ``temperature`` on nonnegative inputs, outputs lie in (0, 1].
        - Enabling ``bias`` and/or non-unit ``temperature`` produces affine/scale-transformed similarities.
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

        # Store configuration flags
        self.learnable_alpha = bool(learnable_alpha)
        self.learnable_beta = bool(learnable_beta)

        # Parameters: weight and optional bias
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
        else:
            self.bias = None

        # Register eps so module.to(device) moves it
        default_dtype = torch.get_default_dtype()
        self.register_buffer("eps", torch.tensor(float(eps), dtype=default_dtype))

        # Alpha/Beta either as learnable parameters (unconstrained) or fixed buffers
        if self.learnable_alpha:
            self._alpha_unconstrained = nn.Parameter(
                torch.tensor(_softplus_inverse(float(alpha)), dtype=default_dtype)
            )
        else:
            self.register_buffer(
                "alpha", torch.tensor(float(alpha), dtype=default_dtype)
            )
        if self.learnable_beta:
            self._beta_unconstrained = nn.Parameter(
                torch.tensor(_softplus_inverse(float(beta)), dtype=default_dtype)
            )
        else:
            self.register_buffer("beta", torch.tensor(float(beta), dtype=default_dtype))

        self.input_transform = input_transform
        self.nonnegative = bool(nonnegative)
        self.smoothing_tau = float(smoothing_tau) if smoothing_tau is not None else None
        self.alpha_beta_normalize = bool(alpha_beta_normalize)
        self.temperature = float(temperature) if temperature is not None else None

        # Initialize parameters similar to nn.Linear
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights and optional bias with Kaiming-uniform heuristics."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward pass (scaffold)."""
        alpha, beta = self._compute_alpha_beta()
        sims = pairwise_tversky(
            x,
            self.weight,
            alpha=float(alpha.item()),
            beta=float(beta.item()),
            eps=float(self.eps.item()),
            input_transform=self.input_transform,
            nonnegative=self.nonnegative,
            smoothing_tau=self.smoothing_tau,
        )
        if self.bias is not None:
            sims = sims + self.bias
        if self.temperature is not None:
            sims = sims * self.temperature
        return sims

    def _compute_alpha_beta(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return current positive (and optionally normalized) alpha and beta.

        When learnable, maps unconstrained parameters through softplus to ensure
        positivity, and optionally normalizes so that alpha + beta = 1.
        """
        if self.learnable_alpha:
            alpha_pos = F.softplus(self._alpha_unconstrained)
        else:
            alpha_pos = self.alpha
        if self.learnable_beta:
            beta_pos = F.softplus(self._beta_unconstrained)
        else:
            beta_pos = self.beta

        if self.alpha_beta_normalize:
            denom = alpha_pos + beta_pos
            # denom should be > 0 due to softplus or positive buffers; guard anyway
            alpha_pos = alpha_pos / denom
            beta_pos = beta_pos / denom
        return alpha_pos, beta_pos


def _softplus_inverse(y: float) -> float:
    """Approximate inverse of softplus for positive y.

    Returns value z such that softplus(z) ~= y. For y>0, a good approximation is
    log(exp(y) - 1).
    """
    # Using expm1 improves numerical stability for small y.
    return math.log(math.expm1(y))
