import inspect

import pytest
import torch

from mlp_utils.layers import (
    MLP,
    NGPT,
    FastFeedForward,
    FeedForward,
    PathWeightedFFF,
    SwitchFFN,
)  # noqa: E402


@pytest.mark.parametrize(
    ("module_ctor", "ctor_kwargs", "expects_tuple"),
    [
        (FeedForward, {"dim": 16}, False),
        (FastFeedForward, {"dim": 16, "depth": 2}, False),
        (PathWeightedFFF, {"input_width": 16, "depth": 2, "output_width": 16}, False),
        (MLP, {"input_dim": 16, "output_dim": 16}, False),
        (NGPT, {"dim": 16}, False),
        (SwitchFFN, {"dim": 16, "num_experts": 4}, True),
    ],
)
def test_feedforward_forward_signature_and_behavior(
    module_ctor, ctor_kwargs, expects_tuple
):
    torch.manual_seed(0)

    batch_size, seq_len, dim = 2, 5, 16
    x = torch.randn(batch_size, seq_len, dim)

    module = module_ctor(**ctor_kwargs)
    module.eval()

    # 1) Verify first non-self forward parameter is named `x`
    sig = inspect.signature(module.forward)
    non_self_params = [p for name, p in sig.parameters.items() if name != "self"]
    assert non_self_params, (
        f"{module_ctor.__name__}.forward should accept at least one argument besides self"
    )
    assert non_self_params[0].name == "x", (
        f"{module_ctor.__name__}.forward's first parameter must be named 'x', got '{non_self_params[0].name}'"
    )

    # 2) Verify calling with keyword `x=` works and output shape matches input shape
    out = module(x=x)

    if expects_tuple:
        assert isinstance(out, tuple) and len(out) == 2, (
            f"{module_ctor.__name__} should return (output, aux_loss)"
        )
        y, aux = out
        assert isinstance(y, torch.Tensor), "Output must be a tensor"
        assert y.shape == x.shape, (
            f"Output shape {y.shape} must match input shape {x.shape}"
        )
        assert isinstance(aux, torch.Tensor) and aux.ndim == 0, (
            "Auxiliary loss must be a 0-D tensor"
        )
    else:
        assert isinstance(out, torch.Tensor), "Output must be a tensor"
        assert out.shape == x.shape, (
            f"Output shape {out.shape} must match input shape {x.shape}"
        )
