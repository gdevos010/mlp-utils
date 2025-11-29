import torch

from mlp_utils.layers import FFNFiLM, FiLMGenerator, ResidualFiLM


def test_residual_film_identity() -> None:
    dim = 12
    module = torch.nn.Identity()
    gen = FiLMGenerator(cond_dim=4, feature_dim=dim, token_wise=False)
    wrapper = ResidualFiLM(module, feature_dim=dim, generator=gen)

    x = torch.randn(2, dim)
    cond = torch.zeros(2, 4)
    y = wrapper(x, cond)
    assert torch.allclose(y, x, atol=1e-6)


def test_ffn_film_shapes() -> None:
    dim = 16
    gen = FiLMGenerator(cond_dim=4, feature_dim=dim * 4, token_wise=False)
    wrapper = FFNFiLM(dim=dim, hidden_mult=4, generator=gen)

    x = torch.randn(3, dim)
    cond = torch.randn(3, 4)
    y = wrapper(x, cond)
    assert y.shape == x.shape
