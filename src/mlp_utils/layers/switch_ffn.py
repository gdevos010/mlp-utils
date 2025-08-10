"""Implements the Switch Transformer Feed-Forward layer.

Based on the paper:
Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
by William Fedus, Barret Zoph, Noam Shazeer.
https://arxiv.org/abs/2101.03961
"""

import torch
import torch.nn.functional as F

from torch import nn

from .feedforward import FeedForward


class SwitchFFN(nn.Module):
    r"""Implements the Switch Transformer Feed-Forward layer.

    A Switch Transformer layer consists of a number of experts, each of which
    is a feed-forward network. For each token, a router decides which expert
    to send the token to. This implementation routes each token to a single
    expert.

    The layer also implements a load balancing loss to encourage all experts
    to be used equally.

    Parameters
    ----------
    dim : int
        Input and output dimension of the layer.
    num_experts : int
        The number of expert networks.
    capacity_factor : float, default 1.25
        Factor to determine the capacity of each expert. The capacity is
        calculated as `(num_tokens / num_experts) * capacity_factor`.
    loss_coef : float, default 1e-2
        Coefficient for the load balancing loss.
    **ff_kwargs : Any
        Keyword arguments to be passed to the `FeedForward` expert networks.
        This can include `mult`, `dropout`, `glu_variant`, etc.

    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        capacity_factor: float = 1.25,
        loss_coef: float = 1e-2,
        ff_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be a positive integer")
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.loss_coef = loss_coef
        self.has_aux_loss = True

        self.router = nn.Linear(dim, num_experts, bias=False)
        _ff_kwargs = ff_kwargs or {}
        self.experts = nn.ModuleList(
            [FeedForward(dim=dim, **_ff_kwargs) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the Switch FFN layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            A tuple containing:
            - torch.Tensor: The output tensor of shape (batch_size, seq_len, dim).
            - torch.Tensor: The auxiliary load balancing loss.
        """
        batch_size, seq_len, dim = x.shape
        x = x.view(-1, dim)
        num_tokens = x.shape[0]

        # Get router logits and probabilities
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(x.dtype)

        # For each token, get the top-1 expert and its gating weight
        gating_weights, expert_indices = torch.max(router_probs, dim=-1)

        # Create a one-hot mask for the expert assignments
        expert_mask = F.one_hot(expert_indices, self.num_experts)

        # Calculate expert capacity and drop tokens that exceed it
        capacity = int((num_tokens / self.num_experts) * self.capacity_factor)
        capacity = max(capacity, 1)

        position_in_expert = torch.cumsum(expert_mask, dim=0) - 1
        capacity_mask = position_in_expert < capacity
        expert_mask = expert_mask * capacity_mask

        # Mask out the gating weights of dropped tokens
        gating_weights = gating_weights * expert_mask.sum(dim=-1)

        # Calculate auxiliary load balancing loss
        tokens_per_expert = expert_mask.sum(dim=0)
        fraction_tokens_per_expert = tokens_per_expert / num_tokens
        fraction_probs_per_expert = router_probs.mean(dim=0)
        load_balancing_loss = (
            self.loss_coef
            * self.num_experts
            * (fraction_tokens_per_expert * fraction_probs_per_expert).sum()
        )

        # Dispatch tokens to their experts
        final_output = torch.zeros_like(x)
        for i, expert_ffn in enumerate(self.experts):
            token_indices = torch.where(expert_mask[:, i] == 1)[0]
            if token_indices.numel() > 0:
                tokens_for_expert = x[token_indices]
                expert_output = expert_ffn(tokens_for_expert)
                weighted_output = expert_output * gating_weights[token_indices, None]
                final_output.index_add_(0, token_indices, weighted_output)

        # For dropped tokens, use the original input (identity function)
        dropped_mask = expert_mask.sum(dim=-1) == 0
        final_output[dropped_mask] = x[dropped_mask]

        final_output = final_output.view(batch_size, seq_len, dim)
        return final_output, load_balancing_loss
