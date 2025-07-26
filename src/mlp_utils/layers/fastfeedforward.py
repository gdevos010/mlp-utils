"""A PyTorch implementation of a fast feedforward network."""

from typing import Literal

import torch

from torch import nn

from .feedforward import FeedForward
from .glu import SwiGLU


class FastFeedForward(nn.Module):
    """Fast Feedforward Network (FFF) layer.

    Implements the architecture from "Fast Feedforward Networks" by Belcak et al.
    (https://arxiv.org/abs/2308.14711). This layer uses a tree-structured
    routing mechanism to select a small subset of neurons (experts) for each
    input token, achieving logarithmic time complexity with respect to the
    number of experts.

    Each input token is routed through a binary tree of depth `depth`. At each
    node, a learned router decides which of the two children to proceed to.
    The leaves of the tree are `FeedForward` networks (experts) that process
    the token.

    Attributes:
    ----------
    dim : int
        Input and output dimension.
    expert_dim : int
        Dimension of the expert networks.
    depth : int
        Depth of the routing tree.
    num_experts : int
        Total number of experts at the leaves of the tree (2**depth).

    Parameters
    ----------
    dim : int
        Input and output dimension of the layer.
    depth : int
        Depth of the routing tree. The number of experts will be 2**depth.
    expert_dim : int, optional
        Dimension of the expert networks. If not provided, it is set to `dim`.
    mult : int, default 4
        Expansion factor for the hidden layer of each expert FeedForward network.
    dropout : float, default 0.0
        Dropout probability for the expert networks.
    activation : type[nn.Module], default nn.GELU
        Activation used for the vanilla MLP path in experts.
    glu_variant : str, default "none"
        GLU variant for the expert FeedForward networks. See `FeedForward` class.
    pre_norm : bool, default False
        If `True`, applies layer normalization before the expert networks.
    norm_layer : type[nn.Module], default nn.RMSNorm
        Normalization layer class to use if `pre_norm` is `True`.
    soft_routing_during_train : bool, default True
        If `True`, uses differentiable soft routing during training.
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
        self.glu_variant = glu_variant

        if self.dim != self.expert_dim:
            self.project_in = nn.Linear(self.dim, self.expert_dim)
            self.project_out = nn.Linear(self.expert_dim, self.dim)
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
        """Forward pass for the FastFeedForward layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        if self.training and self.soft_routing_during_train:
            return self._soft_routing_forward(x)
        return self._hard_routing_forward(x)

    def _soft_routing_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using soft, differentiable routing. Used for training. Each token's output is a weighted average of all experts' outputs."""
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
        """Determines which expert each token is routed to by traversing the tree."""
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
        """Dispatch to the appropriate hard routing implementation."""
        if self._is_swiglu_fast_path_compatible:
            return self._fast_swiglu_hard_routing(x)
        return self._generic_hard_routing(x)

    def _generic_hard_routing(self, x: torch.Tensor) -> torch.Tensor:
        """Generic hard routing that works with any expert type."""
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
        """Optimized hard routing for SwiGLU experts. Each token is processed by a single expert."""
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
