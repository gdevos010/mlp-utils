"""Implements the MLP block of the Normalized Transformer (nGPT) as described in "nGPT: Normalized Transformer with Representation Learning on the Hypersphere".

(https://arxiv.org/html/2410.01131v2)
"""

import torch
import torch.nn.functional as F

from torch import nn


class NGPT(nn.Module):
    """Implements the MLP block of the Normalized Transformer (nGPT) as described in.

    "nGPT: Normalized Transformer with Representation Learning on the Hypersphere".
    (https://arxiv.org/html/2410.01131v2)

    This block takes a feedforward network and wraps it with the nGPT update rule.
    The input and output hidden states are normalized to have unit L2 norm.
    """

    def __init__(
        self,
        feedforward_net: nn.Module,
        dim: int,
        scalar_alpha: bool = False,
        alpha_m_init: float = 0.5,
    ) -> None:
        """Initializes the nGPT MLP block.

        Args:
            feedforward_net (nn.Module): The feedforward network (MLP) to be wrapped.
            dim (int): The feature dimension of the hidden states.
            scalar_alpha (bool): If True, use a single scalar for alpha_m.
                                 If False, use a vector of size `dim`. Defaults to False.
            alpha_m_init (float): Initial value for the learnable parameter alpha_m.
        """
        super().__init__()
        self.feedforward_net = feedforward_net

        if scalar_alpha:
            self.alpha_m = nn.Parameter(torch.tensor(alpha_m_init))
        else:
            self.alpha_m = nn.Parameter(torch.full((dim,), alpha_m_init))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Applies the nGPT MLP block transformation.

        Args:
            h (torch.Tensor): Input tensor of shape (..., dim).
                              It is assumed to be normalized.

        Returns:
            torch.Tensor: Output tensor of shape (..., dim), which is also normalized.
        """
        # h_M <- Norm(MLP(h))
        # The input h is assumed to be normalized.
        h_m = self.feedforward_net(h)
        h_m_norm = F.normalize(h_m, p=2, dim=-1)

        # h <- Norm(h + α_M * (h_M - h))
        # This is a linear interpolation (LERP) step on the hypersphere.
        h_updated = h + self.alpha_m * (h_m_norm - h)
        h_out = F.normalize(h_updated, p=2, dim=-1)

        return h_out
