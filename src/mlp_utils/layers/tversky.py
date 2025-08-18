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
from typing import Literal
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from torch import nn

__all__ = [
    "tversky_similarity",
    "pairwise_tversky",
    "TverskyProjection",
    "tversky_attributions",
    "TverskyFeatureSharing",
    "TverskySimilarityConfig",
    "TverskyProjectionConfig",
    "TverskyFeatureSharingConfig",
]


@dataclass(slots=True)
class TverskySimilarityConfig:
    """Configuration for Tversky similarity computation.

    ABOUTME: Typed config making Tversky similarity behavior explicit and reusable.
    ABOUTME: Used to construct layers or call functions in a consistent manner.
    """

    alpha: float = 0.5
    beta: float = 0.5
    theta: float = 1e-7
    input_transform: str | None = None
    nonnegative: bool = True
    smoothing_tau: float | None = None
    intersection_reduction: Literal["sum", "mean", "product"] = "sum"
    difference_reduction: Literal["ignorematch", "subtractmatch"] = "subtractmatch"
    match_threshold: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "TverskySimilarityConfig":
        return cls(**data)

    def to_kwargs(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "theta": self.theta,
            "input_transform": self.input_transform,
            "nonnegative": self.nonnegative,
            "smoothing_tau": self.smoothing_tau,
            "intersection_reduction": self.intersection_reduction,
            "difference_reduction": self.difference_reduction,
            "match_threshold": self.match_threshold,
        }


@dataclass(slots=True)
class TverskyProjectionConfig:
    """Configuration for :class:`TverskyProjection`.

    ABOUTME: Encapsulates all constructor options to ease reproducibility.
    ABOUTME: Provides utility methods to instantiate a module from config.
    """

    input_dim: int = 1
    output_dim: int = 1
    # Similarity options
    similarity: TverskySimilarityConfig = field(default_factory=TverskySimilarityConfig)
    # Layer options
    bias: bool = False
    learnable_alpha: bool = False
    learnable_beta: bool = False
    alpha_beta_normalize: bool = False
    temperature: float | None = None
    # Initialization: string key or a callable initializer
    prototype_init: (
        Literal["xavier", "kaiming"] | Callable[[torch.Tensor], None]
    ) = "xavier"
    # Optional learnable variants
    learnable_theta: bool = False
    learnable_match_threshold: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "TverskyProjectionConfig":
        sim = data.get("similarity")
        if isinstance(sim, dict):
            data = {**data, "similarity": TverskySimilarityConfig.from_dict(sim)}
        return cls(**data)

    def to_kwargs(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "alpha": self.similarity.alpha,
            "beta": self.similarity.beta,
            "theta": self.similarity.theta,
            "bias": self.bias,
            "input_transform": self.similarity.input_transform,
            "nonnegative": self.similarity.nonnegative,
            "smoothing_tau": self.similarity.smoothing_tau,
            "learnable_alpha": self.learnable_alpha,
            "learnable_beta": self.learnable_beta,
            "alpha_beta_normalize": self.alpha_beta_normalize,
            "temperature": self.temperature,
            "prototype_init": self.prototype_init,
            "intersection_reduction": self.similarity.intersection_reduction,
            "difference_reduction": self.similarity.difference_reduction,
            "match_threshold": self.similarity.match_threshold,
            "learnable_theta": self.learnable_theta,
            "learnable_match_threshold": self.learnable_match_threshold,
        }


@dataclass(slots=True)
class TverskyFeatureSharingConfig:
    """Configuration for :class:`TverskyFeatureSharing` two-stage head.

    ABOUTME: Bundles stage-1 and stage-2 settings with shapes for easy wiring.
    ABOUTME: Keeps all behaviors configurable while preserving safe defaults.
    """

    input_dim: int = 1
    num_features: int = 1
    output_dim: int = 1
    # Stage 1 and 2 similarity options
    s1: TverskySimilarityConfig = field(default_factory=lambda: TverskySimilarityConfig(input_transform="clamp01"))
    s2: TverskySimilarityConfig = field(default_factory=lambda: TverskySimilarityConfig(input_transform="clamp01"))
    # Layer options per stage
    s1_bias: bool = False
    s1_temperature: float | None = None
    s1_prototype_init: Literal["xavier", "kaiming"] = "xavier"
    s2_bias: bool = False
    s2_temperature: float | None = None
    s2_prototype_init: Literal["xavier", "kaiming"] = "xavier"

    @classmethod
    def from_dict(cls, data: dict) -> "TverskyFeatureSharingConfig":
        s1 = data.get("s1")
        s2 = data.get("s2")
        if isinstance(s1, dict):
            data = {**data, "s1": TverskySimilarityConfig.from_dict(s1)}
        if isinstance(s2, dict):
            data = {**data, "s2": TverskySimilarityConfig.from_dict(s2)}
        return cls(**data)

def tversky_similarity(  # noqa: C901, PLR0913, PLR0912, PLR0915
    input: torch.Tensor,  # noqa: A002
    prototype: torch.Tensor,
    *,
    alpha: float | torch.Tensor = 0.5,
    beta: float | torch.Tensor = 0.5,
    theta: float | torch.Tensor = 1e-7,
    input_transform: str | Callable[[torch.Tensor], torch.Tensor] | None = None,
    nonnegative: bool = True,
    smoothing_tau: float | None = None,
    intersection_reduction: Literal["sum", "mean", "product"] = "sum",
    difference_reduction: Literal["ignorematch", "subtractmatch"] = "subtractmatch",
    match_threshold: float = 0.0,
) -> torch.Tensor:
    """Compute the (differentiable) Tversky similarity between two tensors.

    Uses per-feature proxy set operations to estimate intersection and distinct
    parts, then applies the Tversky similarity:

        S = (I + theta) / (I + alpha * A_only + beta * B_only + theta)

    where I = sum(min(x, y)), A_only = sum(relu(x - y)), B_only = sum(relu(y - x)).
    Optional smoothing via `smoothing_tau` replaces hard min and ReLU with smooth
    approximations that approach the hard ops as tau -> 0.

    Args:
        input: Input tensor of shape (..., D).
        prototype: Prototype tensor of shape (..., D), broadcastable with `input`.
        alpha: Weight for input-only mass.
        beta: Weight for prototype-only mass.
        theta: Numerical stability constant added to numerator/denominator.
        input_transform: Optional transform applied before set-op proxies. One of
            None, "relu", "clamp01", "sigmoid", or a callable. If provided, it
            supersedes `nonnegative`.
        nonnegative: If True and no explicit transform is provided, clamp to
            nonnegative via ReLU.
        smoothing_tau: Optional temperature for smoothed proxies; if provided,
            must be > 0.
        intersection_reduction: How to aggregate intersection over features.
            One of {"sum", "mean", "product"}. Default: "sum".
        difference_reduction: How to treat distinctive parts. One of
            {"subtractmatch", "ignorematch"}. "ignorematch" discards difference
            contributions at feature positions where both input and prototype are
            present (> ``match_threshold``).
        match_threshold: Threshold used to determine feature presence when
            ``difference_reduction='ignorematch'``.

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
        i_components = torch.minimum(x, y)
        a_components = F.relu(x - y)
        b_components = F.relu(y - x)
    else:
        tau = float(smoothing_tau)
        # Ensure shapes match for stacking in broadcasting scenarios
        x_b, y_b = torch.broadcast_tensors(x, y)
        # Smooth minimum via soft-min: -tau * logsumexp([-x/tau, -y/tau])
        stacked = torch.stack((x_b, y_b), dim=-1)  # (..., D, 2)
        soft_min = -tau * torch.logsumexp(-stacked / tau, dim=-1)  # (..., D)
        # Add shift so that for nonnegative inputs, intersection components
        # remain nonnegative and match the attribution path.
        i_components = soft_min + tau * math.log(2.0)

        # Smooth ReLU via tau * softplus((z)/tau)
        a_components = tau * F.softplus((x_b - y_b) / tau)
        b_components = tau * F.softplus((y_b - x_b) / tau)

    # Apply difference handling option
    if difference_reduction not in {"subtractmatch", "ignorematch"}:
        raise ValueError(
            "difference_reduction must be one of 'subtractmatch' or 'ignorematch'"
        )
    if difference_reduction == "ignorematch":
        # Ignore difference contributions where both are present beyond threshold
        # Work on broadcast-aligned shapes
        x_b, y_b = torch.broadcast_tensors(x, y)
        both_present = (x_b > match_threshold) & (y_b > match_threshold)
        a_components = a_components * (~both_present).to(a_components.dtype)
        b_components = b_components * (~both_present).to(b_components.dtype)

    # Aggregate across feature dimension
    if intersection_reduction not in {"sum", "mean", "product"}:
        raise ValueError(
            "intersection_reduction must be one of 'sum', 'mean', or 'product'"
        )
    if intersection_reduction == "sum":
        intersection = i_components.sum(dim=-1)
    elif intersection_reduction == "mean":
        intersection = i_components.mean(dim=-1)
    else:  # product
        # Product can underflow quickly; keep as-is but clamp to non-negative
        intersection = i_components.clamp_min(0.0).prod(dim=-1)

    a_only = a_components.sum(dim=-1)
    b_only = b_components.sum(dim=-1)

    similarity = (intersection + theta) / (
        intersection + alpha * a_only + beta * b_only + theta
    )
    return similarity


def pairwise_tversky(  # noqa: PLR0913
    input: torch.Tensor,  # noqa: A002
    prototypes: torch.Tensor,
    *,
    alpha: float | torch.Tensor = 0.5,
    beta: float | torch.Tensor = 0.5,
    theta: float | torch.Tensor = 1e-7,
    input_transform: str | Callable[[torch.Tensor], torch.Tensor] | None = None,
    nonnegative: bool = True,
    smoothing_tau: float | None = None,
    intersection_reduction: Literal["sum", "mean", "product"] = "sum",
    difference_reduction: Literal["ignorematch", "subtractmatch"] = "subtractmatch",
    match_threshold: float = 0.0,
) -> torch.Tensor:
    """Compute pairwise Tversky similarities between inputs and prototypes.

    Vectorizes over the prototype axis and supports broadcasting over all leading
    dimensions of ``input``. Internally delegates to :func:`tversky_similarity`.

    Args:
        input: Tensor of shape ``[..., D]``.
        prototypes: Tensor of shape ``[K, D]`` where each row is a prototype.
        alpha: Weight for input-only mass.
        beta: Weight for prototype-only mass.
        theta: Numerical stability constant.
        input_transform: Optional transform applied before proxy set operations.
        nonnegative: If True and no explicit transform is provided, clamp to nonnegative.
        smoothing_tau: Optional temperature for smoothed proxies; if provided, must be > 0.
        intersection_reduction: Feature aggregation for intersection term.
        difference_reduction: Treatment of distinctive parts.
        match_threshold: Presence threshold for 'ignorematch'.

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
        theta=theta,
        input_transform=input_transform,
        nonnegative=nonnegative,
        smoothing_tau=smoothing_tau,
        intersection_reduction=intersection_reduction,
        difference_reduction=difference_reduction,
        match_threshold=match_threshold,
    )
    return sims


def tversky_attributions(  # noqa: PLR0913
    input: torch.Tensor,  # noqa: A002
    prototype: torch.Tensor,
    *,
    input_transform: str | Callable[[torch.Tensor], torch.Tensor] | None = None,
    nonnegative: bool = True,
    smoothing_tau: float | None = None,
    # alpha/beta are not used to compute components themselves; they are included for API symmetry
    alpha: float = 0.5,
    beta: float = 0.5,
    difference_reduction: Literal["ignorematch", "subtractmatch"] = "subtractmatch",
    match_threshold: float = 0.0,
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
        difference_reduction: If 'ignorematch', zero-out difference components
            where both input and prototype are present (> ``match_threshold``).
        match_threshold: Presence threshold for 'ignorematch'.

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
        # Add shift to keep components nonnegative and consistent with similarity
        i_components = soft_min + tau * math.log(2.0)
        a_components = tau * F.softplus((x_b - y_b) / tau)
        b_components = tau * F.softplus((y_b - x_b) / tau)

    if difference_reduction not in {"subtractmatch", "ignorematch"}:
        raise ValueError(
            "difference_reduction must be one of 'subtractmatch' or 'ignorematch'"
        )
    if difference_reduction == "ignorematch":
        x_b, y_b = torch.broadcast_tensors(x, y)
        both_present = (x_b > match_threshold) & (y_b > match_threshold)
        a_components = a_components * (~both_present).to(a_components.dtype)
        b_components = b_components * (~both_present).to(b_components.dtype)

    return i_components, a_components, b_components


class TverskyProjection(nn.Module):
    """Projection layer using Tversky similarity against learned prototypes.

    Parameters:
        input_dim: Feature dimension of inputs.
        output_dim: Number of prototypes (output channels).
        alpha, beta: Weights for distinctive parts in Tversky denominator. Stored as buffers.
        theta: Numerical stability constant for similarity formula.
        bias: If True, add a learnable bias of shape ``[output_dim]``.
        input_transform: Optional transform applied before proxy set operations.
        nonnegative: If True and no explicit transform is provided, clamp to nonnegative.
        smoothing_tau: Optional temperature for smoothed proxies; if provided, must be > 0.
        learnable_alpha, learnable_beta: Reserved for future extension (not enabled yet).
        alpha_beta_normalize: Reserved for future extension to renormalize alpha/beta.
        temperature: Optional post-similarity scaling factor.
        prototype_init: Prototype weight initialization. One of {"xavier",
            "kaiming"}. Default: "xavier".
        intersection_reduction: Feature aggregation for intersection. One of
            {"sum", "mean", "product"}. Default: "sum".
        difference_reduction: Treatment of distinctive parts. One of
            {"subtractmatch", "ignorematch"}. Default: "subtractmatch".
        match_threshold: Presence threshold for 'ignorematch'.

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
        theta: float = 1e-7,
        bias: bool = False,
        input_transform: str | Callable[[torch.Tensor], torch.Tensor] | None = None,
        nonnegative: bool = True,
        smoothing_tau: float | None = None,
        learnable_alpha: bool = False,
        learnable_beta: bool = False,
        alpha_beta_normalize: bool = False,
        temperature: float | None = None,
        prototype_init: Literal["xavier", "kaiming"] | Callable[[torch.Tensor], None] = "xavier",
        intersection_reduction: Literal["sum", "mean", "product"] = "sum",
        difference_reduction: Literal["ignorematch", "subtractmatch"] = "subtractmatch",
        match_threshold: float = 0.0,
        learnable_theta: bool = False,
        learnable_match_threshold: bool = False,
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

        # Theta: either learnable (unconstrained -> softplus) or fixed buffer
        default_dtype = torch.get_default_dtype()
        self.learnable_theta = bool(learnable_theta)
        if self.learnable_theta:
            self._theta_unconstrained = nn.Parameter(
                torch.tensor(_softplus_inverse(float(theta)), dtype=default_dtype)
            )
        else:
            self.register_buffer("theta", torch.tensor(float(theta), dtype=default_dtype))

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

        # Validate and store options
        # Validate initializer: allow callable or known keys
        init_options = {"xavier", "kaiming"}
        if not (callable(prototype_init) or prototype_init in init_options):
            raise ValueError(
                "prototype_init must be 'xavier', 'kaiming', or a callable initializer"
            )
        if intersection_reduction not in {"sum", "mean", "product"}:
            raise ValueError(
                "intersection_reduction must be one of 'sum', 'mean', or 'product'"
            )
        if difference_reduction not in {"subtractmatch", "ignorematch"}:
            raise ValueError(
                "difference_reduction must be one of 'subtractmatch' or 'ignorematch'"
            )

        self.input_transform = input_transform
        self.nonnegative = bool(nonnegative)
        self.smoothing_tau = float(smoothing_tau) if smoothing_tau is not None else None
        self.alpha_beta_normalize = bool(alpha_beta_normalize)
        self.temperature = float(temperature) if temperature is not None else None
        self.prototype_init = prototype_init
        self.intersection_reduction = intersection_reduction
        self.difference_reduction = difference_reduction
        # Match threshold: optionally learnable nonnegative scalar
        self.learnable_match_threshold = bool(learnable_match_threshold)
        if self.learnable_match_threshold:
            self._match_threshold_unconstrained = nn.Parameter(
                torch.tensor(_softplus_inverse(float(match_threshold)), dtype=default_dtype)
            )
        else:
            self.match_threshold = float(match_threshold)

        # Initialize parameters similar to nn.Linear
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights and optional bias according to configuration."""
        if callable(self.prototype_init):
            try:
                self.prototype_init(self.weight)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("Custom prototype_init callable failed") from exc
        else:
            if self.prototype_init == "xavier":
                nn.init.xavier_uniform_(self.weight)
            else:
                nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward pass producing similarity scores per prototype."""
        alpha, beta = self._compute_alpha_beta()
        theta = self._compute_theta()
        match_threshold = self._compute_match_threshold()
        sims = pairwise_tversky(
            x,
            self.weight,
            alpha=alpha,
            beta=beta,
            theta=theta,
            input_transform=self.input_transform,
            nonnegative=self.nonnegative,
            smoothing_tau=self.smoothing_tau,
            intersection_reduction=self.intersection_reduction,
            difference_reduction=self.difference_reduction,
            match_threshold=match_threshold,
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
            denom = (alpha_pos + beta_pos).clamp_min(torch.finfo(alpha_pos.dtype).eps)
            # denom should be > 0 due to softplus or positive buffers; guard anyway
            alpha_pos = alpha_pos / denom
            beta_pos = beta_pos / denom
        return alpha_pos, beta_pos

    def _compute_theta(self) -> torch.Tensor:
        """Return current positive theta as a scalar tensor."""
        if self.learnable_theta:
            return F.softplus(self._theta_unconstrained)
        # self.theta is a buffer if not learnable
        assert isinstance(self.theta, torch.Tensor)
        return self.theta

    def _compute_match_threshold(self) -> torch.Tensor | float:
        """Return nonnegative match threshold used for 'ignorematch' behavior."""
        if self.learnable_match_threshold:
            return F.softplus(self._match_threshold_unconstrained)
        return self.match_threshold

    @classmethod
    def from_config(cls, cfg: TverskyProjectionConfig) -> "TverskyProjection":
        """Construct a :class:`TverskyProjection` from a config object."""
        return cls(**cfg.to_kwargs())


def _softplus_inverse(y: float) -> float:
    """Approximate inverse of softplus for positive y.

    Returns value z such that softplus(z) ~= y. For y>0, a good approximation is
    log(exp(y) - 1).
    """
    # Using expm1 improves numerical stability for small y.
    return math.log(math.expm1(y))


class TverskyFeatureSharing(nn.Module):
    """Two-stage head with shared features followed by prototype similarities.

    Stage 1 maps input vectors into a shared feature-membership space using a
    first :class:`TverskyProjection`. Stage 2 scores those memberships against
    a bank of feature-space prototypes using a second :class:`TverskyProjection`.

    Shapes:
        - Input: ``[..., D_in]``
        - Stage 1 output (memberships): ``[..., M]`` where ``M=num_features``
        - Final output: ``[..., K]`` where ``K=output_dim``

    Notes:
        - Defaults are chosen to mirror the paper's canonical setup: no bias,
          no temperature, intersection reduced by sum, and differences included.
        - The two stages accept independent hyperparameters for flexibility.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        input_dim: int,
        num_features: int,
        output_dim: int,
        # Stage 1 (data -> features)
        s1_alpha: float = 0.5,
        s1_beta: float = 0.5,
        s1_theta: float = 1e-7,
        s1_input_transform: str
        | Callable[[torch.Tensor], torch.Tensor]
        | None = "clamp01",
        s1_nonnegative: bool = True,
        s1_smoothing_tau: float | None = None,
        s1_intersection_reduction: Literal["sum", "mean", "product"] = "sum",
        s1_difference_reduction: Literal[
            "ignorematch", "subtractmatch"
        ] = "subtractmatch",
        s1_match_threshold: float = 0.0,
        s1_bias: bool = False,
        s1_temperature: float | None = None,
        s1_prototype_init: Literal["xavier", "kaiming"] = "xavier",
        # Stage 2 (features -> prototypes)
        s2_alpha: float = 0.5,
        s2_beta: float = 0.5,
        s2_theta: float = 1e-7,
        s2_input_transform: str
        | Callable[[torch.Tensor], torch.Tensor]
        | None = "clamp01",
        s2_nonnegative: bool = True,
        s2_smoothing_tau: float | None = None,
        s2_intersection_reduction: Literal["sum", "mean", "product"] = "sum",
        s2_difference_reduction: Literal[
            "ignorematch", "subtractmatch"
        ] = "subtractmatch",
        s2_match_threshold: float = 0.0,
        s2_bias: bool = False,
        s2_temperature: float | None = None,
        s2_prototype_init: Literal["xavier", "kaiming"] = "xavier",
    ) -> None:
        super().__init__()

        # Stage 1: input_dim -> num_features
        self.stage1 = TverskyProjection(
            input_dim=input_dim,
            output_dim=num_features,
            alpha=s1_alpha,
            beta=s1_beta,
            theta=s1_theta,
            bias=s1_bias,
            input_transform=s1_input_transform,
            nonnegative=s1_nonnegative,
            smoothing_tau=s1_smoothing_tau,
            temperature=s1_temperature,
            prototype_init=s1_prototype_init,
            intersection_reduction=s1_intersection_reduction,
            difference_reduction=s1_difference_reduction,
            match_threshold=s1_match_threshold,
        )

        # Stage 2: num_features -> output_dim
        self.stage2 = TverskyProjection(
            input_dim=num_features,
            output_dim=output_dim,
            alpha=s2_alpha,
            beta=s2_beta,
            theta=s2_theta,
            bias=s2_bias,
            input_transform=s2_input_transform,
            nonnegative=s2_nonnegative,
            smoothing_tau=s2_smoothing_tau,
            temperature=s2_temperature,
            prototype_init=s2_prototype_init,
            intersection_reduction=s2_intersection_reduction,
            difference_reduction=s2_difference_reduction,
            match_threshold=s2_match_threshold,
        )

    def reset_parameters(self) -> None:
        """Reset parameters of both stages."""
        self.stage1.reset_parameters()
        self.stage2.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Compute membership with Stage 1, then prototype scores with Stage 2."""
        memberships = self.stage1(x)
        return self.stage2(memberships)

    @torch.no_grad()
    def get_feature_memberships(self, x: torch.Tensor) -> torch.Tensor:
        """Return Stage 1 membership vector z(x) in feature space."""
        return self.stage1(x)

    def freeze_features(self, freeze: bool = True) -> None:
        """Enable/disable gradient updates for Stage 1 parameters."""
        for param in self.stage1.parameters():
            param.requires_grad_(not freeze)

    def freeze_head(self, freeze: bool = True) -> None:
        """Enable/disable gradient updates for Stage 2 parameters."""
        for param in self.stage2.parameters():
            param.requires_grad_(not freeze)

    @classmethod
    def from_config(
        cls, cfg: TverskyFeatureSharingConfig
    ) -> "TverskyFeatureSharing":
        """Construct from a :class:`TverskyFeatureSharingConfig`."""
        return cls(
            input_dim=cfg.input_dim,
            num_features=cfg.num_features,
            output_dim=cfg.output_dim,
            # Stage 1
            s1_alpha=cfg.s1.alpha,
            s1_beta=cfg.s1.beta,
            s1_theta=cfg.s1.theta,
            s1_input_transform=cfg.s1.input_transform,
            s1_nonnegative=cfg.s1.nonnegative,
            s1_smoothing_tau=cfg.s1.smoothing_tau,
            s1_intersection_reduction=cfg.s1.intersection_reduction,
            s1_difference_reduction=cfg.s1.difference_reduction,
            s1_match_threshold=cfg.s1.match_threshold,
            s1_bias=cfg.s1_bias,
            s1_temperature=cfg.s1_temperature,
            s1_prototype_init=cfg.s1_prototype_init,
            # Stage 2
            s2_alpha=cfg.s2.alpha,
            s2_beta=cfg.s2.beta,
            s2_theta=cfg.s2.theta,
            s2_input_transform=cfg.s2.input_transform,
            s2_nonnegative=cfg.s2.nonnegative,
            s2_smoothing_tau=cfg.s2.smoothing_tau,
            s2_intersection_reduction=cfg.s2.intersection_reduction,
            s2_difference_reduction=cfg.s2.difference_reduction,
            s2_match_threshold=cfg.s2.match_threshold,
            s2_bias=cfg.s2_bias,
            s2_temperature=cfg.s2_temperature,
            s2_prototype_init=cfg.s2_prototype_init,
        )
