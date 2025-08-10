"""A PyTorch implementation of a fast feedforward network."""

from typing import Literal

import torch

from torch import nn

from .feedforward import FeedForward
from .glu import SwiGLU


class FastFeedForward(nn.Module):
    """Fast Feedforward Network (FFF) layer.

    Implements the architecture from "Fast Feedforward Networks" by Belcak & Wattenhofer
    (`https://arxiv.org/abs/2308.14711`). A binary routing tree selects one expert per
    token during hard routing; soft routing mixes leaf experts during training if enabled.

    This layer expects 3D inputs and preserves the leading two dimensions.

    Args:
        dim (int): Input and output model dimension.
        depth (int): Tree depth; number of experts is `2 ** depth`.
        expert_dim (int | None): Expert model dimension. Defaults to `dim`.
        mult (int): Expansion factor for expert hidden layers. Defaults to 4.
        dropout (float): Dropout probability inside experts. Defaults to 0.0.
        activation (type[nn.Module]): Activation for vanilla MLP experts. Defaults to `nn.GELU`.
        glu_variant (Literal["none","glu","geglu","swiglu","reglu","bilinear",
            "mglu","mgeglu","mswiglu","mreglu","mbilinear"]): Expert type. Defaults to "none".
        pre_norm (bool): Apply `norm_layer` before expert nets. Defaults to False.
        norm_layer (type[nn.Module]): Normalization class when `pre_norm` is True. Defaults to `nn.RMSNorm`.
        soft_routing_during_train (bool): Use differentiable soft routing while training. Defaults to True.
    """

    def __init__(  # noqa: PLR0913, C901
        self,
        dim: int,
        depth: int,
        expert_dim: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
        glu_variant: Literal[
            "none",
            "glu",
            "geglu",
            "swiglu",
            "reglu",
            "bilinear",
            "mglu",
            "mgeglu",
            "mswiglu",
            "mreglu",
            "mbilinear",
        ] = "none",
        pre_norm: bool = False,
        norm_layer: type[nn.Module] = nn.RMSNorm,
        soft_routing_during_train: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("Tree depth must be at least 1.")

        self.dim = dim
        self.expert_dim = expert_dim if expert_dim is not None else dim
        self.depth = depth
        self.num_experts = 2**depth
        self.soft_routing_during_train = soft_routing_during_train
        self.glu_variant = glu_variant.lower()

        if self.dim != self.expert_dim:
            self.project_in: nn.Module = nn.Linear(self.dim, self.expert_dim)
            self.project_out: nn.Module = nn.Linear(self.expert_dim, self.dim)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

        # Routers: one for each internal node of the binary tree
        # Total internal nodes = 2**depth - 1
        num_routers = self.num_experts - 1
        self.routers = nn.ModuleList([nn.Linear(dim, 1) for _ in range(num_routers)])

        # Experts: one at each leaf of the tree
        self.experts = nn.ModuleList(
            [
                FeedForward(
                    dim=self.expert_dim,
                    mult=mult,
                    dropout=dropout,
                    activation=activation,
                    glu_variant=glu_variant,
                    pre_norm=pre_norm,
                    norm_layer=norm_layer,
                )
                for _ in range(self.num_experts)
            ]
        )

        self._is_swiglu_fast_path_compatible = self._check_swiglu_compatibility()

    def _check_swiglu_compatibility(self) -> bool:
        """Check if the experts are compatible with the fast SwiGLU path."""
        if self.glu_variant != "swiglu":
            return False

        first_expert = self.experts[0]
        if not isinstance(first_expert, FeedForward):
            return False
        if not hasattr(first_expert, "net") or not isinstance(
            first_expert.net, nn.Sequential
        ):
            return False
        if len(first_expert.net) < 3:  # noqa: PLR2004
            return False

        if not isinstance(first_expert.net[0], SwiGLU):
            return False
        return isinstance(first_expert.net[2], nn.Linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Fast Feedforward layer.

        Args:
            x (torch.Tensor): Input of shape (batch_size, seq_len, dim) and floating dtype.

        Returns:
            torch.Tensor: Output of shape (batch_size, seq_len, dim) with the same dtype as `x`.
        """
        if self.training and self.soft_routing_during_train:
            return self._soft_routing_forward(x)
        return self._hard_routing_forward(x)

    def _soft_routing_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Soft, differentiable routing (training).

        Each token's output is a weighted average of leaf expert outputs.

        Args:
            x (torch.Tensor): Input of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output of shape (batch_size, seq_len, dim).
        """
        batch_size, seq_len, dim = x.shape
        flat_x = x.reshape(batch_size * seq_len, dim)
        num_tokens = flat_x.shape[0]

        # --- Soft Routing ---
        # Calculate path probabilities for each token to each leaf expert.
        leaf_probs = torch.ones(num_tokens, 1, device=x.device)

        for d in range(self.depth):
            router_offset = 2**d - 1
            num_routers_at_level = 2**d
            level_routers = self.routers[
                router_offset : router_offset + num_routers_at_level
            ]

            w = torch.cat([r.weight for r in level_routers], dim=0)
            b = torch.cat([r.bias for r in level_routers], dim=0)

            routing_logits = torch.einsum("nd, rd -> nr", flat_x, w) + b
            routing_probs = torch.sigmoid(routing_logits)

            # (num_tokens, 2**d, 1)
            leaf_probs = leaf_probs.unsqueeze(2)
            # (num_tokens, 2**d, 1)
            probs_left = 1 - routing_probs.unsqueeze(2)
            probs_right = routing_probs.unsqueeze(2)

            # (num_tokens, 2**d, 2)
            combined_probs = torch.cat([probs_left, probs_right], dim=2)

            # Update leaf probabilities by multiplying by branch probabilities
            leaf_probs = leaf_probs * combined_probs
            # (num_tokens, 2**(d+1))
            leaf_probs = leaf_probs.view(num_tokens, -1)

        # --- Expert Computation ---
        # Get output from all experts for all tokens.
        projected_x = self.project_in(flat_x)
        all_expert_outputs = torch.stack(
            [expert(projected_x) for expert in self.experts], dim=1
        )

        # Weight expert outputs by the calculated probabilities and sum them up.
        output = torch.einsum("nep,ne->np", all_expert_outputs, leaf_probs)
        output = self.project_out(output)
        return output.reshape(batch_size, seq_len, dim)

    def _get_expert_indices(self, flat_x: torch.Tensor) -> torch.Tensor:
        """Traverse the routing tree to select a leaf expert per token.

        Args:
            flat_x (torch.Tensor): Flattened inputs of shape (num_tokens, dim).

        Returns:
            torch.Tensor: Integer tensor of shape (num_tokens,) with expert indices in [0, num_experts).
        """
        num_tokens = flat_x.shape[0]
        leaf_indices = torch.zeros(num_tokens, dtype=torch.long, device=flat_x.device)

        for d in range(self.depth):
            router_offset = 2**d - 1
            num_routers_at_level = 2**d
            level_routers = self.routers[
                router_offset : router_offset + num_routers_at_level
            ]

            w = torch.cat([r.weight for r in level_routers], dim=0)
            b = torch.cat([r.bias for r in level_routers], dim=0)

            routing_logits = torch.einsum("nd, rd -> nr", flat_x, w) + b

            current_node_indices = leaf_indices.unsqueeze(1)
            selected_logits = torch.gather(
                routing_logits, 1, current_node_indices
            ).squeeze(1)

            decision = (selected_logits > 0).long()
            leaf_indices = leaf_indices * 2 + decision

        return leaf_indices

    def _hard_routing_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Hard routing forward pass.

        Dispatches to an optimized path when compatible, otherwise uses a generic path.
        """
        if self._is_swiglu_fast_path_compatible:
            return self._fast_swiglu_hard_routing(x)
        return self._generic_hard_routing(x)

    def _generic_hard_routing(self, x: torch.Tensor) -> torch.Tensor:
        """Generic hard routing compatible with any expert type.

        Args:
            x (torch.Tensor): Input of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output of shape (batch_size, seq_len, dim).
        """
        batch_size, seq_len, dim = x.shape
        flat_x = x.reshape(batch_size * seq_len, dim)

        # --- Routing ---
        leaf_indices = self._get_expert_indices(flat_x)

        # --- Expert Computation ---
        projected_x = self.project_in(flat_x)
        sorted_indices = torch.argsort(leaf_indices)
        restore_indices = torch.argsort(sorted_indices)

        sorted_x = projected_x[sorted_indices]
        sorted_leaf_indices = leaf_indices[sorted_indices]

        output_chunks = []
        counts = torch.bincount(sorted_leaf_indices, minlength=self.num_experts)
        start_idx = 0
        for i, count in enumerate(counts):
            if count == 0:
                continue
            end_idx = start_idx + count
            expert_input = sorted_x[start_idx:end_idx]
            expert_output = self.experts[i](expert_input)
            output_chunks.append(expert_output)
            start_idx = end_idx

        if output_chunks:
            output_states = torch.cat(output_chunks, dim=0)
        else:
            output_states = torch.empty_like(sorted_x)

        unsorted_output = output_states[restore_indices]
        output = self.project_out(unsorted_output)
        return output.reshape(batch_size, seq_len, dim)

    def _fast_swiglu_hard_routing(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized hard routing for SwiGLU experts.

        Processes each token with a single expert using batched matmuls.

        Args:
            x (torch.Tensor): Input of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output of shape (batch_size, seq_len, dim).
        """
        batch_size, seq_len, dim = x.shape
        flat_x = x.reshape(batch_size * seq_len, dim)

        # --- Routing ---
        leaf_indices = self._get_expert_indices(flat_x)

        # --- Expert Computation ---
        # `leaf_indices` now holds the final expert index for each token.

        # Stack weights for batched matrix multiplication
        # The SwiGLU layer combines w1 and w2 into a single projection
        w1_w2 = torch.stack([expert.net[0].proj.weight for expert in self.experts])
        b1_b2 = torch.stack([expert.net[0].proj.bias for expert in self.experts])
        # The final projection layer is the third element in the sequential module
        w3 = torch.stack([expert.net[2].weight for expert in self.experts])
        b3 = torch.stack([expert.net[2].bias for expert in self.experts])

        # Sort tokens by expert index to improve memory access patterns.
        sorted_indices = torch.argsort(leaf_indices)
        restore_indices = torch.argsort(sorted_indices)

        projected_x = self.project_in(flat_x)
        sorted_x = projected_x[sorted_indices]
        sorted_leaf_indices = leaf_indices[sorted_indices]

        # Gather the appropriate expert weights for each token.
        w1_w2_per_token = w1_w2[sorted_leaf_indices]
        b1_b2_per_token = b1_b2[sorted_leaf_indices]
        w3_per_token = w3[sorted_leaf_indices]
        b3_per_token = b3[sorted_leaf_indices]

        # Batched FeedForward (SwiGLU) logic for each token.
        # Unsqueeze sorted_x for batched matrix multiplication (bmm).
        hidden_and_gated = torch.bmm(
            sorted_x.unsqueeze(1), w1_w2_per_token.transpose(1, 2)
        ).squeeze(1)
        hidden_and_gated = hidden_and_gated + b1_b2_per_token

        hidden_states, gated_states = hidden_and_gated.chunk(2, dim=-1)
        activated_states = torch.nn.functional.silu(hidden_states) * gated_states

        output_states = torch.bmm(
            activated_states.unsqueeze(1), w3_per_token.transpose(1, 2)
        ).squeeze(1)
        output_states = output_states + b3_per_token

        # Un-sort to restore original token order
        unsorted_output = output_states[restore_indices]

        output = self.project_out(unsorted_output)
        return output.reshape(batch_size, seq_len, dim)
