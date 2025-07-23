from typing import Literal

import torch

from torch import nn

from .feedforward import FeedForward


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
        self.depth = depth
        self.num_experts = 2**depth
        self.soft_routing_during_train = soft_routing_during_train

        # Routers: one for each internal node of the binary tree
        # Total internal nodes = 2**depth - 1
        num_routers = self.num_experts - 1
        self.routers = nn.ModuleList([nn.Linear(dim, 1) for _ in range(num_routers)])

        # Experts: one at each leaf of the tree
        self.experts = nn.ModuleList(
            [
                FeedForward(
                    dim=dim,
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
        """
        Forward pass using soft, differentiable routing. Used for training.
        Each token's output is a weighted average of all experts' outputs.
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
        all_expert_outputs = torch.stack(
            [expert(flat_x) for expert in self.experts], dim=1
        )

        # Weight expert outputs by the calculated probabilities and sum them up.
        output = torch.einsum("nep,ne->np", all_expert_outputs, leaf_probs)
        return output.reshape(batch_size, seq_len, dim)

    def _hard_routing_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using hard, non-differentiable routing. Used for inference.
        Each token is processed by a single expert.
        """
        batch_size, seq_len, dim = x.shape
        flat_x = x.reshape(batch_size * seq_len, dim)
        num_tokens = flat_x.shape[0]

        # --- Routing ---
        # Determine the expert index for each token by traversing the tree.
        # `leaf_indices` will store the chosen expert for each token.
        leaf_indices = torch.zeros(num_tokens, dtype=torch.long, device=x.device)

        for d in range(self.depth):
            # The decision at each level refines the path.
            # `router_offset` is the index of the first router at the current depth.
            router_offset = 2**d - 1
            # `num_routers_at_level` is the number of routers at this depth.
            num_routers_at_level = 2**d

            # Stack the weights and biases of all routers at the current level
            # for efficient batch processing.
            level_routers = self.routers[
                router_offset : router_offset + num_routers_at_level
            ]
            w = torch.cat([r.weight for r in level_routers], dim=0)
            b = torch.cat([r.bias for r in level_routers], dim=0)

            # Get routing logits for each token against each router at this level.
            # Shape: (num_tokens, num_routers_at_level)
            routing_logits = torch.einsum("nd, rd -> nr", flat_x, w) + b

            # For each token, select the logit from the router corresponding to
            # its current path in the tree (`leaf_indices`).
            # `gather` selects the appropriate logit for each token.
            current_node_indices = leaf_indices.unsqueeze(1)
            selected_logits = torch.gather(
                routing_logits, 1, current_node_indices
            ).squeeze(1)

            # Make a binary decision (go right if > 0, left if <= 0).
            decision = (selected_logits > 0).long()

            # Update leaf_indices to reflect the path taken.
            # If at node `i`, the children are `2*i` and `2*i + 1`.
            leaf_indices = leaf_indices * 2 + decision

        # --- Expert Computation ---
        # `leaf_indices` now holds the final expert index for each token.
        # We process tokens in batches based on their assigned expert.
        output = torch.zeros_like(flat_x)
        for i in range(self.num_experts):
            expert_mask = leaf_indices == i
            if not expert_mask.any():
                continue

            expert_tokens = flat_x[expert_mask]
            expert_output = self.experts[i](expert_tokens)
            output[expert_mask] = expert_output

        return output.reshape(batch_size, seq_len, dim)
