# ABOUTME: Unit tests for explain_similarity helper.
# ABOUTME: Validates shape and sum consistency with tversky_similarity.

import torch

from mlp_utils.layers.tversky_explain import explain_similarity
from mlp_utils.layers import tversky_similarity


def test_explain_similarity_shapes_and_consistency() -> None:
    torch.manual_seed(0)
    x = torch.rand(3, 8)
    p = torch.rand(8)
    out = explain_similarity(x, p, input_transform="relu")
    assert out["I_components"].shape == x.shape
    assert out["A_components"].shape == x.shape
    assert out["B_components"].shape == x.shape
    assert out["I"].shape == (3,)
    assert out["A"].shape == (3,)
    assert out["B"].shape == (3,)
    s_ref = tversky_similarity(x, p, input_transform="relu")
    assert torch.allclose(out["similarity"], s_ref, rtol=1e-5, atol=1e-6)
