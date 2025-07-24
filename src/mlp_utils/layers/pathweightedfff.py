"""A PyTorch implementation of a Path-Weighted Fast Feedforward Network.

largely inspired by https://github.com/pbelcak/UltraFastBERT/blob/main/benchmark_pytorch/fff/fff_bmm.py"""

import math

import torch

from torch import nn
from torch.autograd.function import Function, FunctionCtx


class SignSTE(Function):
    """Straight-through estimator for the sign function."""

    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.sign(x)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass."""
        return grad_output


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    """Sign function with straight-through estimator."""
    return SignSTE.apply(x)


class PathWeightedFFF(nn.Module):
    """Path-Weighted Fast Feedforward Network (FFF).

    This module implements a hierarchical, path-dependent neural network.
    It uses a binary tree structure to process input tensors. For each input,
    a path is determined from the root to a leaf node by a series of routing
    decisions.

    Unlike a standard Mixture-of-Experts (MoE) model that selects a single
    expert, this network computes its output by combining transformations from
    *every* node along the traversed path. The routing logits themselves, after
    a GELU activation, act as weights for these transformations.

    This architecture allows the model to learn hierarchical features, where each
    level of the tree can contribute to the final output in a path-dependent
    manner.

    Attributes:
    ----------
    input_width : int
        The dimensionality of the input features.
    depth : int
        The depth of the binary tree. The total number of nodes is 2**(depth+1) - 1.
    output_width : int
        The dimensionality of the output features.
    n_nodes : int
        The total number of nodes in the tree.

    Parameters
    ----------
    input_width : int
        The dimensionality of the input features.
    depth : int
        The depth of the binary tree.
    output_width : int
        The dimensionality of the output features.
    """

    def __init__(
        self,
        input_width: int,
        depth: int,
        output_width: int,
    ) -> None:
        super().__init__()

        if depth < 0:
            raise ValueError("Tree depth must be non-negative.")

        self.input_width = input_width
        self.depth = depth
        self.output_width = output_width

        # Total nodes in a complete binary tree of depth `d` is 2**(d+1) - 1
        self.n_nodes = 2 ** (depth + 1) - 1
        self._initialise_weights()

    def _initialise_weights(self) -> None:
        """Initializes the routing and output weights."""
        init_factor_l1 = 1.0 / math.sqrt(self.input_width)
        init_factor_l2 = 1.0 / math.sqrt(self.input_width)

        self.w1s = nn.Parameter(
            torch.empty(self.n_nodes, self.input_width).uniform_(
                -init_factor_l1, +init_factor_l1
            )
        )
        self.b1s = nn.Parameter(torch.zeros(self.n_nodes))

        self.w2s = nn.Parameter(
            torch.empty(self.n_nodes, self.input_width, self.output_width).uniform_(
                -init_factor_l2, +init_factor_l2
            )
        )
        self.b2s = nn.Parameter(torch.zeros(self.n_nodes, self.output_width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the PathWeightedFFF.

        Can handle both 2D (batch_size, input_width) and 3D
        (batch_size, seq_len, input_width) inputs.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with the same dimensions as the input,
                      but with the last dimension as `output_width`.
        """
        # Handle 3D input by flattening and reshaping
        is_3d = x.dim() == 3
        if is_3d:
            batch_size, seq_len, _ = x.shape
            flat_x = x.reshape(batch_size * seq_len, self.input_width)
        else:
            batch_size = x.shape[0]
            flat_x = x

        if flat_x.dim() != 2 or flat_x.shape[1] != self.input_width:
            raise ValueError(
                f"Input tensor must be of shape (batch_size, input_width) or "
                f"(batch_size, seq_len, input_width), but got shape {x.shape}"
            )

        num_tokens = flat_x.shape[0]
        device = x.device

        # --- Tree Traversal and Logit Collection ---
        # `current_nodes` tracks the active node for each item in the batch.
        # It starts at the root (node 0).
        current_nodes = torch.zeros((num_tokens,), dtype=torch.long, device=device)

        # `all_nodes` will store the full path for each batch item.
        # `all_logits` will store the routing score at each step of the path.
        all_nodes = torch.zeros(
            num_tokens, self.depth + 1, dtype=torch.long, device=device
        )
        all_logits = torch.empty(
            num_tokens, self.depth + 1, dtype=torch.float, device=device
        )

        for i in range(self.depth + 1):
            all_nodes[:, i] = current_nodes

            # Select the routing weights for the current nodes.
            # `w1s` defines the hyperplane for the routing decision.
            plane_coeffs = self.w1s.index_select(dim=0, index=current_nodes)
            bias = self.b1s.index_select(dim=0, index=current_nodes)
            # (num_tokens, input_width)

            # Project input onto the hyperplane to get the routing score (logit).
            plane_score = (
                torch.bmm(flat_x.unsqueeze(1), plane_coeffs.unsqueeze(-1))
                .squeeze(-1)
                .squeeze(-1)
            ) + bias
            # (num_tokens,)
            all_logits[:, i] = plane_score

            # Make the routing decision. Go right if score >= 0, left otherwise.
            # Left child of node `n` is `2n+1`, right is `2n+2`.
            # Use a straight-through estimator for the sign function for differentiability.
            plane_choices = (
                (sign_ste(plane_score) + 1) / 2
            ).long()  # 0 for left, 1 for right
            current_nodes = current_nodes * 2 + plane_choices + 1  # (num_tokens,)

        # routing direction (left or right).
        path_weights = torch.nn.functional.gelu(torch.abs(all_logits))

        # --- Path-Weighted Output Computation ---
        # The final output is a weighted sum of the outputs of transformations at each node
        # along the path. Each node applies a linear transformation (w2, b2) to the input.
        output = torch.zeros(
            num_tokens, self.output_width, dtype=x.dtype, device=x.device
        )

        for i in range(self.depth + 1):
            # Get the nodes at the current depth for all items in the batch
            nodes_at_depth = all_nodes[:, i]

            # Get the weights and biases for these nodes
            # w2_i: (num_tokens, input_width, output_width)
            # b2_i: (num_tokens, output_width)
            w2_i = self.w2s.index_select(0, nodes_at_depth)
            b2_i = self.b2s.index_select(0, nodes_at_depth)

            # Apply the linear transformation
            # (num_tokens, 1, input_width) @ (num_tokens, input_width, output_width)
            # -> (num_tokens, 1, output_width) -> (num_tokens, output_width)
            node_transform = torch.bmm(flat_x.unsqueeze(1), w2_i).squeeze(1) + b2_i

            # Get the path weights for the current depth
            # path_weights shape: (num_tokens, depth + 1)
            current_path_weights = path_weights[:, i].unsqueeze(1)  # (num_tokens, 1)

            # Add the weighted, transformed output of the current nodes to the total output
            output += node_transform * current_path_weights

        # Reshape back to original dimensions if input was 3D
        if is_3d:
            output = output.view(batch_size, seq_len, self.output_width)

        return output
