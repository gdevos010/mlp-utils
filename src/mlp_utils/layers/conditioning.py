"""ABOUTME: Conditioning wrappers that apply FiLM around modules.
ABOUTME: Provides ResidualFiLM and FFNFiLM wrappers and generator factories.
"""

from collections.abc import Callable
from typing import Any

import torch

from torch import nn

from .film import FiLM, FiLMGenerator


class ResidualFiLM(nn.Module):
    """Pre-norm residual wrapper with FiLM conditioning.

    Applies y = x + module( FiLM( norm(x), cond ) ) where FiLM is applied to the
    normalized input and also modulates the residual stream.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        feature_dim: int,
        generator: FiLMGenerator
        | Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.module = module
        self.norm = norm_layer(feature_dim)
        self.film = FiLM(feature_dim=feature_dim)
        self.generator = generator

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM-conditioned residual update.

        Args:
            x: Input tensor of shape (..., feature_dim).
            cond: Conditioning tensor used to compute FiLM parameters.

        Returns:
            Tensor with residual FiLM modulation applied.
        """
        h = self.norm(x)
        # Baseline path without FiLM for identity at zero-conditioning
        baseline = self.module(h)
        gamma, beta = self.generator(cond)
        h_mod = self.film(h, gamma, beta)
        modulated = self.module(h_mod)
        return x + (modulated - baseline)


class FFNFiLM(nn.Module):
    """Applies FiLM to the intermediate activation of an FFN-like module.

    Expects `ffn` to be a sequential or a module exposing a hidden activation
    point via a callable hook. For simplicity, we wrap a standard 2-layer MLP.
    """

    def __init__(
        self,
        dim: int,
        *,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
        generator: FiLMGenerator
        | Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        super().__init__()
        hidden = dim * hidden_mult
        self.in_proj = nn.Linear(dim, hidden)
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden, dim)
        self.film = FiLM(feature_dim=hidden)
        self.generator = generator

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM to FFN hidden activations and project back.

        Args:
            x: Input tensor of shape (..., dim).
            cond: Conditioning tensor used to compute FiLM parameters.

        Returns:
            Tensor after FiLM-modulated FFN.
        """
        h = self.in_proj(x)
        gamma, beta = self.generator(cond)
        h = self.film(h, gamma, beta)
        h = self.act(h)
        h = self.dropout(h)
        return self.out_proj(h)


def build_film_generators(
    *,
    shared: bool,
    num_layers: int,
    factory: Callable[..., FiLMGenerator],
    **kwargs: Any,  # noqa: ANN401
) -> FiLMGenerator | list[FiLMGenerator]:
    """Create one shared FiLMGenerator or a list (per-layer).

    Args:
        shared: If True, return a single shared generator instance.
        num_layers: Number of layers for per-layer creation when shared=False.
        factory: Callable that constructs a FiLMGenerator with kwargs.
        **kwargs: Passed to factory.
    """
    if shared:
        return factory(**kwargs)
    return [factory(**kwargs) for _ in range(num_layers)]
