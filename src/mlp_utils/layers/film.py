"""ABOUTME: Core FiLM modules for feature-wise linear modulation and generators.
ABOUTME: Provides FiLM, FiLMGenerator, and LowRankFiLM for conditioning models.
"""

import torch

from torch import nn


class FiLM(nn.Module):
    """Applies feature-wise linear modulation to an activation tensor.

    Computes y = (1 + gamma) ⊙ x + beta, broadcasting gamma/beta over all
    non-feature dimensions. The last dimension is treated as the feature axis.

    Args:
        feature_dim (int): Size of the last (feature) dimension of the input.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim

    def _broadcast_to_input(self, param: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Broadcasts `param` to match `x` across all non-feature axes.

        Accepts shapes [..., D] or [..., 1, D] (already broadcastable) and will
        insert singleton dimensions before the last axis until ranks match.
        """
        if param.shape[-1] != self.feature_dim:
            raise ValueError(
                f"FiLM parameter last dimension must equal feature_dim={self.feature_dim}, "
                f"got {param.shape}"
            )

        out = param
        while out.dim() < x.dim():
            out = out.unsqueeze(-2)
        return out

    def forward(
        self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            x: Input tensor of shape (..., feature_dim).
            gamma: Scale offsets of shape (broadcastable to x), last dim feature_dim.
            beta: Shift offsets of shape (broadcastable to x), last dim feature_dim.

        Returns:
            Modulated tensor of the same shape as `x`.
        """
        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Input last dimension must equal feature_dim={self.feature_dim}, got {x.shape}"
            )

        gamma_b = self._broadcast_to_input(gamma, x)
        beta_b = self._broadcast_to_input(beta, x)
        return x * (1.0 + gamma_b) + beta_b


class FiLMGenerator(nn.Module):
    """Maps a conditioning signal to FiLM parameters (gamma, beta).

    Ensures zero input maps to zero output by using bias-free linear layers.

    Args:
        cond_dim (int): Dimension of the conditioning vector per token or global.
        feature_dim (int): Target feature dimension for FiLM parameters.
        token_wise (bool): If True, expects [B, T, cond_dim] and produces
            [B, T, feature_dim] outputs; otherwise operates on [B, cond_dim].
        hidden (Optional[int]): Optional hidden size for a 2-layer MLP; if None,
            uses a single linear projection to 2 * feature_dim.
        activation (type[nn.Module]): Activation for the hidden layer when used.
    """

    def __init__(
        self,
        *,
        cond_dim: int,
        feature_dim: int,
        token_wise: bool = False,
        hidden: int | None = None,
        activation: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        layers: list[nn.Module] = []
        if hidden is not None and hidden > 0:
            layers.append(nn.Linear(cond_dim, hidden, bias=False))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden, 2 * feature_dim, bias=False))
        else:
            layers.append(nn.Linear(cond_dim, 2 * feature_dim, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate FiLM parameters from the conditioning signal.

        Supports cond shapes [B, C] or [B, T, C]. Returns (gamma, beta) with
        matching leading dimensions and last dimension = feature_dim.
        """
        out = self.net(cond)
        gamma, beta = out.split(self.feature_dim, dim=-1)
        return gamma, beta


class LowRankFiLM(nn.Module):
    """Low-rank FiLM using learned bases for gamma and beta.

    Let coeffs = [c_gamma, c_beta] with shapes (..., rank) each. Then:
      delta_gamma = c_gamma @ basis_gamma  -> (..., feature_dim)
      beta        = c_beta @ basis_beta    -> (..., feature_dim)
    y = (1 + delta_gamma) ⊙ x + beta

    Args:
        feature_dim (int): Size of the last (feature) dimension.
        rank (int): Low-rank size for the bases.
    """

    def __init__(self, *, feature_dim: int, rank: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.rank = rank
        # Initialize near-zero so zero coeffs yield identity and small coeffs are stable.
        self.basis_gamma = nn.Parameter(torch.zeros(rank, feature_dim))
        self.basis_beta = nn.Parameter(torch.zeros(rank, feature_dim))

    def _broadcast_coeffs(self, coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = coeffs
        while out.dim() < x.dim():
            out = out.unsqueeze(-2)
        return out

    def forward(self, x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """Apply low-rank FiLM.

        Args:
            x: Activation tensor of shape (..., feature_dim).
            coeffs: Concatenated coefficients of shape (..., 2 * rank).
        """
        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Input last dimension must equal feature_dim={self.feature_dim}, got {x.shape}"
            )

        c_gamma, c_beta = coeffs.split(self.rank, dim=-1)
        c_gamma = self._broadcast_coeffs(c_gamma, x)
        c_beta = self._broadcast_coeffs(c_beta, x)

        # Compute linear combinations along the rank axis.
        # Shapes: (..., rank) @ (rank, D) -> (..., D)
        delta_gamma = torch.einsum("...r,rd->...d", c_gamma, self.basis_gamma)
        beta = torch.einsum("...r,rd->...d", c_beta, self.basis_beta)
        return x * (1.0 + delta_gamma) + beta


def film_l2_regularization(
    gamma: torch.Tensor,
    beta: torch.Tensor,
    *,
    gamma_weight: float = 1.0,
    beta_weight: float = 1.0,
) -> torch.Tensor:
    """Computes an L2 penalty encouraging FiLM parameters to stay near zero.

    Use this optional term in the training loss to stabilize conditioning and
    preserve identity behavior at initialization.

    Args:
        gamma: FiLM scale offsets with last dimension = feature_dim.
        beta: FiLM shift offsets with last dimension = feature_dim.
        gamma_weight: Weight for the gamma penalty term.
        beta_weight: Weight for the beta penalty term.

    Returns:
        Scalar tensor equal to gamma_weight * ||gamma||^2_mean + beta_weight * ||beta||^2_mean.
    """
    return gamma_weight * gamma.pow(2).mean() + beta_weight * beta.pow(2).mean()
