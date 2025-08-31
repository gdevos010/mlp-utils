import torch

from mlp_utils.layers.film import FiLM, FiLMGenerator, LowRankFiLM

BATCH = 3
TOKENS = 5
DIM = 8
COND = 6


def test_film_identity_global() -> None:
    x = torch.randn(BATCH, TOKENS, DIM)
    cond = torch.zeros(BATCH, COND)

    gen = FiLMGenerator(cond_dim=COND, feature_dim=DIM, token_wise=False)
    film = FiLM(feature_dim=DIM)

    gamma, beta = gen(cond)
    y = film(x, gamma, beta)

    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)


def test_film_identity_token_wise() -> None:
    x = torch.randn(BATCH, TOKENS, DIM)
    cond = torch.zeros(BATCH, TOKENS, COND)

    gen = FiLMGenerator(cond_dim=COND, feature_dim=DIM, token_wise=True)
    film = FiLM(feature_dim=DIM)

    gamma, beta = gen(cond)
    y = film(x, gamma, beta)

    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)


def test_film_broadcasting_global_to_tokens() -> None:
    x = torch.randn(BATCH, TOKENS, DIM)
    cond = torch.randn(BATCH, COND)

    gen = FiLMGenerator(cond_dim=COND, feature_dim=DIM, token_wise=False)
    film = FiLM(feature_dim=DIM)

    gamma, beta = gen(cond)  # [B, D]
    y = film(x, gamma, beta)  # broadcast to [B, T, D]

    assert y.shape == x.shape


def test_film_grad_flow() -> None:
    x = torch.randn(BATCH, TOKENS, DIM)
    cond = torch.randn(BATCH, COND)

    gen = FiLMGenerator(cond_dim=COND, feature_dim=DIM, token_wise=False)
    film = FiLM(feature_dim=DIM)

    gamma, beta = gen(cond)
    y = film(x, gamma, beta)
    loss = y.pow(2).mean()
    loss.backward()

    grad_norm = sum(p.grad.abs().sum() for p in gen.parameters() if p.grad is not None)
    assert grad_norm > 0


def test_lowrank_film_shapes_and_identity() -> None:
    x = torch.randn(BATCH, TOKENS, DIM)
    coeffs = torch.zeros(BATCH, TOKENS, 2 * 4)  # rank=4

    lr_film = LowRankFiLM(feature_dim=DIM, rank=4)
    y = lr_film(x, coeffs)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)
