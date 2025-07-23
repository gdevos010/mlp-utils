from torch import nn

from .init_weights import initialize_weights


def create_mlp(  # noqa
    input_dim: int,
    output_dim: int | None = None,
    hidden_factor: int = 4,
    dropout: float = 0.1,
    act_fn: type[nn.Module] = nn.GELU,
    use_norm: bool = True,
    norm_layer: type[nn.Module] = nn.RMSNorm,
    residual: bool = False,
    pre_norm: bool = False,
) -> nn.Module:
    """Create a standardized MLP module.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension (defaults to input_dim if None)
        hidden_factor: Multiplier for hidden dimension (hidden_dim = input_dim * hidden_factor)
        dropout: Dropout probability
        act_fn: Activation function
        use_norm: Whether to use normalization layers
        norm_layer: Normalization layer class to use
        residual: Whether to add a residual connection
        pre_norm: Whether to apply normalization before or after the first linear layer

    Returns:
        Sequential MLP module
    """
    if hidden_factor <= 0:
        raise ValueError(f"hidden_factor must be positive, got {hidden_factor}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0.0,1.0), got {dropout}")

    if output_dim is None:
        output_dim = input_dim

    hidden_dim = input_dim * hidden_factor
    layers = []

    # Pre-normalization if enabled
    if pre_norm and use_norm:
        layers.append(norm_layer(input_dim))

    # First linear layer and normalization
    layers.append(nn.Linear(input_dim, hidden_dim))
    if use_norm and not pre_norm:
        layers.append(norm_layer(hidden_dim))

    # Handle both class and instance activations
    if isinstance(act_fn, type):
        layers.append(act_fn())
    else:
        layers.append(act_fn)

    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    # Output layer and normalization
    layers.append(nn.Linear(hidden_dim, output_dim))
    if use_norm:
        layers.append(norm_layer(output_dim))

    # Create the model
    model = nn.Sequential(*layers)

    for name, module in model.named_modules():
        initialize_weights(module, layer_name=name)

    # If using residual connection and input_dim == output_dim
    if residual and input_dim == output_dim:
        return ResidualWrapper(model)

    return model
