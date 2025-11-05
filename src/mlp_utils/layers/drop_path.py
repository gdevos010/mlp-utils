import torch

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
