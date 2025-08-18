# ABOUTME: Example script training a tiny TverskyProjection on XOR.
# ABOUTME: Demonstrates usage, determinism, and quick CPU-only convergence.

"""Minimal XOR training example for `TverskyProjection`.

This script trains a single `TverskyProjection` layer on a repeated XOR dataset
and prints final accuracy and elapsed time. It is CPU-only and runs quickly.
"""

from __future__ import annotations

import argparse
import time

import torch

from mlp_utils.layers import TverskyProjection


def make_xor_dataset(repeats: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a small XOR dataset repeated ``repeats`` times.

    Args:
        repeats: Number of times to repeat the 4 XOR patterns.

    Returns:
        Tuple of tensors ``(features, labels)``.
    """
    feats = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    labels = torch.tensor([0, 1, 1, 0])
    feats = feats.repeat(repeats, 1)
    labels = labels.repeat(repeats)
    return feats, labels


def train_xor(
    steps: int = 200,
    lr: float = 0.1,
    seed: int = 1234,
    optimizer: str = "adam",
) -> tuple[float, float]:
    """Train for a small number of steps and return (accuracy, elapsed_seconds)."""
    torch.manual_seed(seed)
    feats, labels = make_xor_dataset()

    # Use a slightly overcomplete prototype bank followed by a linear readout
    model = torch.nn.Sequential(
        TverskyProjection(
            2,
            4,
            input_transform="clamp01",
            nonnegative=True,
            bias=True,
        ),
        torch.nn.Linear(4, 2, bias=True),
    )

    # Prototype initialization at XOR corners to break symmetry and speed up learning
    with torch.no_grad():
        proj: TverskyProjection = model[0]  # type: ignore[assignment]
        corner_prototypes = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=proj.weight.dtype,
        )
        proj.weight.copy_(corner_prototypes)
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    for _ in range(steps):
        logits = model(feats)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    elapsed = time.time() - start

    with torch.no_grad():
        preds = model(feats).argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
    return acc, elapsed


def main() -> None:
    """Entry point: parse args, train, and print summary."""
    parser = argparse.ArgumentParser(description="TverskyProjection XOR example")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--opt", type=str, choices=["adam", "sgd"], default="adam")
    args = parser.parse_args()

    acc, elapsed = train_xor(steps=args.steps, lr=args.lr, optimizer=args.opt)
    print(
        f"Final accuracy: {acc:.3f} | Elapsed: {elapsed:.3f}s | steps={args.steps}, lr={args.lr}, opt={args.opt}"
    )


if __name__ == "__main__":
    main()
