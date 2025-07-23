"""Standardized weight initialization utilities for neural network layers.

This module provides a centralized set of utilities for initializing weights
across different types of neural network layers, ensuring consistency throughout
the codebase.
"""

import math

from collections.abc import Callable
from typing import Literal

from torch import nn


def initialize_weights(  # noqa
    module: nn.Module,
    init_method: Literal[
        "default", "attention", "gating", "embedding", "expert"
    ] = "default",
    nonlinearity: str = "relu",
    scale: float = 1.0,
    layer_name: str = "",
) -> None:
    """Initialize weights of a neural network module based on its type and role.

    Args:
        module: The module to initialize
        init_method: Initialization strategy to use
            - default: Standard initialization for general layers
            - attention: For attention layers (Q, K, V projections)
            - gating: For gates with sigmoid outputs (bias initialized to 1.0)
            - embedding: For embedding layers
            - expert: For expert networks in MoE
        nonlinearity: Nonlinearity used after the layer (for Kaiming init)
        scale: Scale factor to apply to the initialization
        layer_name: Name of the layer (for conditional initialization)
    """
    # Detect layer role from name if method is "default"
    if init_method == "default":
        if any(x in layer_name.lower() for x in ["gate", "gating"]):
            init_method = "gating"
        elif any(
            x in layer_name.lower() for x in ["q_proj", "k_proj", "v_proj", "attention"]
        ):
            init_method = "attention"
        elif any(x in layer_name.lower() for x in ["embed", "token"]):
            init_method = "embedding"
        elif any(x in layer_name.lower() for x in ["expert", "router"]):
            init_method = "expert"

    # Linear layers
    if isinstance(module, nn.Linear):
        if init_method == "attention":
            # For attention projections (Q, K, V)
            nn.init.xavier_uniform_(module.weight, gain=1 / math.sqrt(2) * scale)
        elif init_method == "gating":
            # For gating mechanisms
            nn.init.xavier_uniform_(module.weight, gain=0.1 * scale)
            if module.bias is not None:
                nn.init.constant_(module.bias, 1.0)
        elif init_method == "embedding":
            # For embedding projections
            bound = 1 / math.sqrt(module.weight.size(0))
            nn.init.uniform_(module.weight, -bound * scale, bound * scale)
        elif init_method == "expert":
            # For expert networks (slightly smaller init)
            nn.init.kaiming_normal_(
                module.weight, mode="fan_in", nonlinearity=nonlinearity
            )
            nn.init.zeros_(module.bias) if module.bias is not None else None
        else:
            # Default initialization
            nn.init.kaiming_normal_(
                module.weight, mode="fan_in", nonlinearity=nonlinearity
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # Convolutional layers
    elif isinstance(module, nn.Conv1d | nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight, mode="fan_out", nonlinearity=nonlinearity
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    # Batch normalization and layer normalization
    elif isinstance(
        module, nn.BatchNorm1d | nn.BatchNorm2d | nn.LayerNorm | nn.RMSNorm
    ):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)

    # Embedding layers
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02 * scale)

    # GRU/LSTM layers
    elif isinstance(module, nn.GRU | nn.LSTM):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


def apply_initialization(
    model: nn.Module,
    init_method: Literal[
        "default", "attention", "gating", "embedding", "expert"
    ] = "default",
    nonlinearity: str = "relu",
    scale: float = 1.0,
) -> None:
    """Apply initialization to all modules in a model.

    Args:
        model: The model to initialize
        init_method: Initialization strategy to use (see initialize_weights)
        nonlinearity: Nonlinearity used after the layer (for Kaiming init)
        scale: Scale factor to apply to the initialization
    """
    for name, module in model.named_modules():
        initialize_weights(
            module=module,
            init_method=init_method,
            nonlinearity=nonlinearity,
            scale=scale,
            layer_name=name,
        )


def init_attention_blocks(
    attention_module: nn.Module, scale: float = 1.0, init_output_near_zero: bool = True
) -> None:
    """Initialize attention blocks with specialized attention initialization.

    Args:
        attention_module: The attention module to initialize
        scale: Scale factor for initialization
        init_output_near_zero: Whether to initialize output projection near zero
    """
    # Initialize query/key/value projections
    for name, module in attention_module.named_modules():
        if any(x in name for x in ["q_proj", "k_proj", "v_proj"]):
            if "lime" in name:
                continue
            nn.init.xavier_uniform_(module.weight, gain=1 / math.sqrt(2) * scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # Initialize output projection with small weights
        elif "output_proj" in name or "out_proj" in name:
            if init_output_near_zero:
                nn.init.xavier_uniform_(module.weight, gain=0.1 * scale)
            else:
                nn.init.xavier_uniform_(module.weight, gain=1.0 * scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def create_initializer(
    init_method: Literal["default", "attention", "gating", "embedding", "expert"],
    nonlinearity: str = "relu",
    scale: float = 1.0,
) -> Callable[[nn.Module], None]:
    """Create a customized initializer function with fixed parameters.

    Args:
        init_method: Initialization strategy to use
        nonlinearity: Nonlinearity used after the layer (for Kaiming init)
        scale: Scale factor to apply to the initialization

    Returns:
        An initializer function that can be applied to a module
    """

    def initializer(module: nn.Module) -> None:
        initialize_weights(
            module=module,
            init_method=init_method,
            nonlinearity=nonlinearity,
            scale=scale,
        )

    return initializer
