"""A PyTorch implementation of gMLP."""

import torch

from einops import rearrange
from torch import nn


class DropPath(nn.Module):
    """Stochastic Depth (per-sample DropPath).

    Randomly drops entire residual paths during training.

    Args:
        drop_prob: Probability of dropping the path.
        scale_by_keep: If True, scales by keep probability to preserve expectation.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = bool(scale_by_keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Apply DropPath.

        Shapes:
            - Input: [B, ...]
            - Output: [B, ...]
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor = random_tensor.div(keep_prob)
        return x * random_tensor


class SpatialGatingUnit(nn.Module):
    """Spatial Gating Unit (SGU) for gMLP.

    Performs a learned linear projection across the sequence dimension on a
    normalized representation, enabling cross-token communication.

    Args:
        dim: Feature dimension of each token (D).
        seq_len: Sequence length (N), also the projection dimension.
        multiply_input: If True, returns ``x * gate(x)`` (current variant).
            If False, returns only the spatial gate ``gate(x)`` (canonical gMLP gate).
        gate_activation: Optional activation applied to the gate output. Defaults
            to ``nn.Identity()``.

    Shapes:
        - Input: ``[batch_size, seq_len, dim]`` = ``[B, N, D]``
        - Output: ``[B, N, D]``
    """

    def __init__(
        self,
        dim: int,
        seq_len: int,
        *,
        multiply_input: bool = True,
        gate_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.proj = nn.Linear(seq_len, seq_len)
        self.multiply_input = multiply_input
        self.gate_activation = (
            gate_activation if gate_activation is not None else nn.Identity()
        )
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Spatial Gating Unit.

        Args:
            x: Tensor of shape ``[B, N, D]``.

        Returns:
            Tensor of shape ``[B, N, D]``.
        """
        if x.shape[1] != self.proj.in_features:
            raise ValueError(
                f"SGU seq_len mismatch: expected N={self.proj.in_features}, got N={x.shape[1]}"
            )

        shortcut = x
        x_norm = self.norm(x)
        gate = rearrange(x_norm, "b n d -> b d n")
        gate = self.proj(gate)
        gate = rearrange(gate, "b d n -> b n d")
        gate = self.gate_activation(gate)
        if self.multiply_input:
            return shortcut * gate
        return gate


class GMLPBlock(nn.Module):
    """Single gMLP block.

    Applies a GLU-style split with a Spatial Gating Unit (SGU) to enable cross-token
    communication.

    Args:
        dim: Input and output model dimension (D).
        dim_ff: Feedforward inner (hidden) dimension.
        seq_len: Sequence length (N) used by the SGU.
        canonical_gate: If True, use canonical gMLP gating ``u * gate(v)``.
            If False, use the existing variant ``u * (v * gate(v))``.

    Shapes:
        - Input: ``[B, N, D]``
        - Output: ``[B, N, D]``
    """

    def __init__(
        self,
        dim: int,
        dim_ff: int,
        seq_len: int,
        *,
        canonical_gate: bool = False,
        gate_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.proj_in = nn.Linear(dim, dim_ff * 2)
        self.activation = nn.GELU()

        # When canonical_gate=True, SGU returns only the gate; otherwise it returns v * gate(v)
        self.sgu = SpatialGatingUnit(
            dim_ff,
            seq_len,
            multiply_input=not canonical_gate,
            gate_activation=gate_activation,
        )
        self.proj_out = nn.Linear(dim_ff, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gMLP block.

        Args:
            x (torch.Tensor): Input of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output of shape (batch_size, seq_len, dim).
        """
        u, v = self.proj_in(x).chunk(2, dim=-1)
        u = self.activation(u)
        v = self.sgu(v)
        x = u * v
        x = self.proj_out(x)
        return x


class GMLP(nn.Module):
    """gMLP stack.

    A sequence of pre-norm residual gMLP blocks.

    Args:
        dim: Token feature dimension (D).
        dim_ff: Inner (hidden) dimension for each block.
        seq_len: Sequence length (N).
        depth: Number of stacked blocks.
        canonical_gate: If True, use canonical gMLP gating in all blocks; otherwise
            use the existing variant for backward-compatibility.
        drop_path: Stochastic depth rate applied to the residual branch. Defaults
            to ``0.0`` (disabled).

    Shapes:
        - Input: ``[B, N, D]``
        - Output: ``[B, N, D]``
    """

    def __init__(
        self,
        dim: int,
        dim_ff: int,
        seq_len: int,
        depth: int,
        *,
        canonical_gate: bool = False,
        drop_path: float = 0.0,
        gate_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.RMSNorm(dim),
                    GMLPBlock(
                        dim=dim,
                        dim_ff=dim_ff,
                        seq_len=seq_len,
                        canonical_gate=canonical_gate,
                        gate_activation=gate_activation,
                    ),
                )
                for _ in range(depth)
            ]
        )
        # Per-block DropPath so stochastic depth is applied on each residual branch
        self.drop_paths = nn.ModuleList(
            [
                DropPath(drop_path) if drop_path and drop_path > 0.0 else nn.Identity()
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gMLP stack.

        Args:
            x (torch.Tensor): Input of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output of shape (batch_size, seq_len, dim).
        """
        for block, drop in zip(self.blocks, self.drop_paths, strict=False):
            residual = x
            x = drop(block(x)) + residual
        return x
