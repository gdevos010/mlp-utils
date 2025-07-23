import torch
import torch.nn.functional as F

from mlp_utils.activations import (
    BSiLU,
    NeLU,
    ReluNelu,
    ReluSquared,
    StraightThroughEstimator,
    Gelu2,
    SugarReLU,
)


def test_relu_squared_unsigned() -> None:
    activation = ReluSquared()
    input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected_output = torch.tensor([0.0, 0.0, 0.0, 1.0, 4.0])
    output_tensor = activation(input_tensor)
    assert torch.allclose(output_tensor, expected_output)


def test_relu_squared_signed() -> None:
    activation = ReluSquared(signed=True)
    input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected_output = torch.tensor([-0.0, -0.0, 0.0, 1.0, 4.0])
    output_tensor = activation(input_tensor)
    assert torch.allclose(output_tensor, expected_output)


def test_gelu2() -> None:
    activation = Gelu2()
    input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected_output = F.gelu(input_tensor).pow(2)
    output_tensor = activation(input_tensor)
    assert torch.allclose(output_tensor, expected_output)


def test_bsilu() -> None:
    activation = BSiLU()
    input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    alpha = 1.67
    expected_output = (input_tensor + alpha) * torch.sigmoid(input_tensor) - alpha / 2
    output_tensor = activation(input_tensor)
    assert torch.allclose(output_tensor, expected_output)


def test_nelu() -> None:
    alpha = 0.05
    activation = NeLU(alpha=alpha)
    input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected_output = -alpha / (1.0 + input_tensor.square())
    output_tensor = activation(input_tensor)
    assert torch.allclose(output_tensor, expected_output)


def test_straight_through_estimator() -> None:
    input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    ste = StraightThroughEstimator(F.relu, torch.sigmoid)
    output = ste(input_tensor)

    # Test forward pass
    expected_forward = F.relu(input_tensor)
    assert torch.allclose(output, expected_forward)

    # Test backward pass
    output.sum().backward()
    # The gradient of the output is the gradient of the backward_fn
    # The gradient of sigmoid is sigmoid(x) * (1 - sigmoid(x))
    expected_grad = torch.sigmoid(input_tensor)
    grad_of_sigmoid = expected_grad * (1 - expected_grad)
    assert torch.allclose(input_tensor.grad, grad_of_sigmoid)


def test_relu_nelu() -> None:
    alpha = 0.05
    input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    activation = ReluNelu(alpha=alpha)
    output = activation(input_tensor)
    # Forward pass should be ReLU
    assert torch.allclose(output, F.relu(input_tensor))
    output.sum().backward()

    # Backward pass should be NeLU for x <= 0 and ReLU for x > 0
    with torch.no_grad():
        nelu_grad = 2 * alpha * input_tensor / (1 + input_tensor.square()).square()
        relu_grad = torch.ones_like(input_tensor)
        expected_grad = torch.where(input_tensor > 0, relu_grad, nelu_grad)

    assert torch.allclose(input_tensor.grad, expected_grad)


def test_sugar_relu() -> None:
    input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    activation = SugarReLU()
    output = activation(input_tensor)
    assert torch.allclose(output, F.relu(input_tensor))
    output.sum().backward()
    expected_grad = torch.sigmoid(input_tensor)
    expected_grad_of_grad = expected_grad * (1 - expected_grad)
    assert torch.allclose(input_tensor.grad, expected_grad_of_grad)
