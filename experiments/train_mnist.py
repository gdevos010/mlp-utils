"""MNIST training runner for MLP-family variants.

This script trains multiple backbone variants from `mlp_utils` on the MNIST
classification task and summarizes results in a rich table. It mirrors the
grouping of configurations used in `verify_training.py`.

Key characteristics:
- Minimal CLI
- Graceful stopping with partial results reported
- No model checkpointing
- Rich table summary at the end

All functions and modules use Google-style docstrings and document tensor
shapes/sizes where applicable.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import logging
import math
import re
import time

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split

# Backbones and utilities (kept consistent with verify_training.py)
from mlp_utils.activations import BSiLU, Gelu2, ReluNelu, ReluSquared
from mlp_utils.layers.fastfeedforward import FastFeedForward
from mlp_utils.layers.feedforward import FeedForward
from mlp_utils.layers.gmlp import GMLP
from mlp_utils.layers.mlp import MLP
from mlp_utils.layers.ngpt import NGPT
from mlp_utils.layers.pathweightedfff import PathWeightedFFF
from mlp_utils.layers.switch_ffn import SwitchFFN


@dataclass
class TrainConfig:
    """Configuration for a single training run.

    Attributes:
        model_name: Backbone type identifier.
        dim: Feature dimension for the backbone. Input patches are projected to this size.
        seq_len: Sequence length (number of patches), S = (28 / P)^2.
        batch_size: Minibatch size for training and evaluation.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.
        patch_size: Patch side length P. Must evenly divide 28.
        # Variant-specific keys (optional):
        act_fn: Activation class or callable for `MLP`.
        residual: Whether to enable residual connections in `MLP`.
        use_norm: Whether to use normalization in `MLP`.
        pre_norm: Legacy flag mapped to `norm_mode='pre'` for `MLP`.
        glu_variant: GLU variant string for FeedForward-family.
        activation: Activation for vanilla FeedForward (when glu_variant='none').
        expert_dim: Expert hidden dimension for `FastFeedForward`.
        depth: Tree depth for `FastFeedForward` or `PathWeightedFFF`.
        num_experts: Expert count for `SwitchFFN`.
        ff_kwargs: Keyword args for experts inside `SwitchFFN`.
        scalar_alpha: nGPT interpolation parameter type (bool for scalar vs. vector).
    """

    model_name: str
    dim: int
    seq_len: int
    batch_size: int
    epochs: int
    lr: float
    patch_size: int
    # Optional/variant-specific
    act_fn: nn.Module | None = None
    residual: bool | None = None
    use_norm: bool | None = None
    pre_norm: bool | None = None
    glu_variant: str | None = None
    activation: nn.Module | None = None
    expert_dim: int | None = None
    depth: int | None = None
    num_experts: int | None = None
    ff_kwargs: dict | None = None
    scalar_alpha: bool | None = None


def patchify_images(images: Tensor, patch_size: int) -> Tensor:
    """Converts images into a sequence of flattened patches.

    Args:
        images: Input images of shape [B, 1, 28, 28], dtype float32, range ~[0, 1].
        patch_size: Patch side length P. Must divide 28.

    Returns:
        patches: Flattened patches of shape [B, S, P^2], where S = (28 / P)^2.
    """
    if (
        images.ndim != 4
        or images.shape[1] != 1
        or images.shape[2] != 28
        or images.shape[3] != 28
    ):
        raise ValueError(
            f"Expected images with shape [B, 1, 28, 28], got {tuple(images.shape)}"
        )
    if 28 % patch_size != 0:
        raise ValueError("patch_size must evenly divide 28")

    batch_size = images.shape[0]
    patches_h = 28 // patch_size
    patches_w = 28 // patch_size
    # [B, 1, patches_h, patches_w, P, P]
    patches = images.unfold(dimension=2, size=patch_size, step=patch_size).unfold(
        dimension=3, size=patch_size, step=patch_size
    )
    # Move patch dims to the end and flatten patch pixels
    # -> [B, patches_h, patches_w, 1, P, P]
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    # -> [B, patches_h * patches_w, P * P]
    patches = patches.reshape(
        batch_size, patches_h * patches_w, patch_size * patch_size
    )
    return patches


class MNISTClassifier(nn.Module):
    """Classifier wrapper around a sequence backbone for MNIST.

    The wrapper performs patchification, projects patches to the model dimension,
    applies a backbone operating on sequences, mean-pools over the sequence
    dimension, and finally classifies into 10 classes.

    Shapes:
        - Input images: [B, 1, 28, 28]
        - After patchify: [B, S, P^2], S = (28 / P)^2
        - After input projection: [B, S, D]
        - Backbone output: [B, S, D]
        - Pooled features: [B, D]
        - Logits: [B, 10]
    """

    def __init__(
        self, backbone: nn.Module, dim: int, patch_size: int, num_classes: int = 10
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Linear(patch_size * patch_size, dim)
        self.classifier = nn.Linear(dim, num_classes)
        self.patch_size = patch_size

    def forward(self, images: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            images: Input images of shape [B, 1, 28, 28].

        Returns:
            logits: Tensor of shape [B, 10], or
            (logits, aux_loss): If the backbone returns an auxiliary loss.
        """
        patches = patchify_images(images, patch_size=self.patch_size)  # [B, S, P^2]
        tokens = self.input_proj(patches)  # [B, S, D]
        backbone_out = self.backbone(tokens)  # [B, S, D] or ([B, S, D], aux_loss)

        if isinstance(backbone_out, tuple):
            sequence_features, aux_loss = backbone_out
        else:
            sequence_features, aux_loss = backbone_out, None

        pooled = sequence_features.mean(dim=1)  # [B, D]
        logits = self.classifier(pooled)  # [B, 10]

        if aux_loss is not None:
            return logits, aux_loss
        return logits


def get_model(config: dict) -> nn.Module:
    """Instantiates a backbone model based on the provided configuration.

    The returned module consumes sequences of shape [B, S, D] and returns
    sequences of the same shape. Some models (e.g., SwitchFFN) may return an
    auxiliary loss along with the sequence output.

    Args:
        config: Dictionary with keys like `model_name`, `dim`, `seq_len`, and
            variant-specific hyperparameters.

    Returns:
        A `torch.nn.Module` backbone operating on [B, S, D].
    """
    model_name = config["model_name"]
    dim = config["dim"]
    seq_len = config["seq_len"]

    if model_name == "mlp":
        norm_mode = "pre" if config.get("pre_norm", False) else "post"
        return MLP(
            input_dim=dim,
            output_dim=dim,
            hidden_factor=4,
            act_fn=config["act_fn"],
            residual=config.get("residual", False),
            use_norm=config.get("use_norm", True),
            norm_mode=norm_mode,
        )

    if model_name == "feedforward":
        return FeedForward(
            dim=dim,
            mult=4,
            glu_variant=config["glu_variant"],
            activation=config.get("activation", nn.GELU),
        )

    if model_name == "fastfeedforward":
        return FastFeedForward(
            dim=dim,
            depth=3,
            mult=4,
            glu_variant=config["glu_variant"],
        )

    if model_name == "pathweightedfff":
        return PathWeightedFFF(input_width=dim, depth=config["depth"], output_width=dim)

    if model_name == "ngpt":
        ff_net = FeedForward(dim=dim, mult=4, glu_variant="swiglu")
        return NGPT(
            feedforward_net=ff_net, dim=dim, scalar_alpha=config["scalar_alpha"]
        )

    if model_name == "gmlp":
        # Optional enhancements: gate activation, canonical gating, and DropPath
        gate_act = config.get("gate_activation")
        if gate_act is not None and isinstance(gate_act, type):
            gate_act = gate_act()
        return GMLP(
            dim=dim,
            dim_ff=dim * 4,
            seq_len=seq_len,
            depth=4,
            canonical_gate=config.get("canonical_gate", False),
            drop_path=config.get("drop_path", 0.0),
            gate_activation=gate_act,
        )

    if model_name == "switch_ffn":
        return SwitchFFN(
            dim=dim, num_experts=config["num_experts"], ff_kwargs=config["ff_kwargs"]
        )

    raise ValueError(f"Unknown model: {model_name}")


def get_model_size(module: nn.Module) -> str:
    """Calculates the number of trainable parameters in a module.

    Args:
        module: Any `torch.nn.Module`.

    Returns:
        Human-readable parameter count, e.g., "33.41K" or "1.23M".
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if num_params >= 1_000_000:
        return f"{num_params / 1e6:.2f}M"
    if num_params >= 1_000:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def format_param_count(count: int) -> str:
    """Formats a parameter count as a human-readable string.

    Args:
        count: Number of parameters.

    Returns:
        String like "33.41K" or "1.23M".
    """
    if count >= 1_000_000:
        return f"{count / 1e6:.2f}M"
    if count >= 1_000:
        return f"{count / 1e3:.2f}K"
    return str(int(count))


def get_num_params(module: nn.Module) -> int:
    """Returns the number of trainable parameters.

    Args:
        module: Any `torch.nn.Module`.

    Returns:
        Integer count of parameters that require gradients.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def parse_param_budgets(budgets_arg: str) -> list[int]:
    """Parses a comma-separated string of parameter budgets.

    Accepts integers or values with K/M suffixes (case-insensitive), e.g.,
    "200K, 0.8M, 150000" -> [200_000, 800_000, 150_000].

    Args:
        budgets_arg: Comma-separated budgets string. Empty string yields [].

    Returns:
        List of integer budgets (number of parameters).
    """
    if not budgets_arg:
        return []
    budgets: list[int] = []
    for token in budgets_arg.split(","):
        token = token.strip()
        if not token:
            continue
        m = re.fullmatch(r"(?i)\s*([0-9]*\.?[0-9]+)\s*([kKmM]?)\s*", token)
        if not m:
            raise SystemExit(f"Invalid budget token: '{token}'")
        value = float(m.group(1))
        suffix = m.group(2).lower()
        if suffix == "k":
            value *= 1_000
        elif suffix == "m":
            value *= 1_000_000
        budgets.append(int(round(value)))
    return budgets


def estimate_dim_for_budget(
    base_config: dict,
    variant_config: dict,
    target_params: int,
    tolerance: float = 0.05,
    min_dim: int = 4,
    max_dim: int = 2048,
) -> tuple[int, int]:
    """Finds a `dim` that approximately matches a parameter budget.

    The search constructs the `MNISTClassifier` with the given variant and
    varies `dim` until the total number of trainable parameters (including
    input projection and classifier) falls within the tolerance band.

    Args:
        base_config: Shared config providing `patch_size`, `epochs`, etc.
        variant_config: Model-specific config overrides (e.g., `model_name`).
        target_params: Desired parameter count (total, including the wrapper).
        tolerance: Fractional tolerance, e.g., 0.05 for ±5%.
        min_dim: Lower bound for dimension search (inclusive).
        max_dim: Upper bound for dimension search (inclusive).

    Returns:
        (chosen_dim, actual_params): The selected dimension and the realized
        parameter count.
    """

    def count_for_dim(dim: int) -> int:
        cfg = {**base_config, **variant_config}
        cfg["dim"] = int(dim)
        # Ensure seq_len matches current patch size
        patch_size_local = cfg["patch_size"]
        cfg["seq_len"] = (28 // patch_size_local) ** 2
        backbone_local = get_model(cfg)
        model_local = MNISTClassifier(
            backbone=backbone_local, dim=cfg["dim"], patch_size=patch_size_local
        )
        return get_num_params(model_local)

    # Compute base params at the base dim for an initial guess.
    base_dim = int(base_config.get("dim", 128))
    try:
        base_params = count_for_dim(base_dim)
    except Exception:
        # Fallback if a particular variant cannot instantiate at base dim
        base_params = max(1, target_params)

    # Heuristic initial guess based on ~quadratic scaling in dim.
    scale = max(1e-6, target_params / max(1, base_params))
    guess = int(max(min_dim, min(max_dim, round(base_dim * math.sqrt(scale)))))

    # Expand bounds to bracket the target if needed.
    low, high = min_dim, max_dim
    # Tighten low/high around guess using exponential search for efficiency.
    lo = max(min_dim, guess)
    hi = max(min_dim, guess)
    params_at_guess = count_for_dim(guess)
    if params_at_guess < target_params:
        while hi < max_dim:
            hi = min(max_dim, hi * 2)
            p = count_for_dim(hi)
            if p >= target_params:
                break
        low, high = lo, hi
    else:
        while lo > min_dim:
            lo = max(min_dim, lo // 2)
            p = count_for_dim(lo)
            if p <= target_params:
                break
        low, high = lo, max(lo + 1, hi)

    # Binary search to meet tolerance band.
    lower_bound = int(target_params * (1.0 - tolerance))
    upper_bound = int(target_params * (1.0 + tolerance))
    best_dim = guess
    best_err = abs(params_at_guess - target_params)
    best_params = params_at_guess
    for _ in range(24):
        mid = int((low + high) // 2)
        mid = int(max(min_dim, min(max_dim, mid)))
        p = count_for_dim(mid)
        # Track best
        err = abs(p - target_params)
        if err < best_err:
            best_err = err
            best_dim = mid
            best_params = p
        if lower_bound <= p <= upper_bound:
            best_dim = mid
            best_params = p
            break
        if p < target_params:
            low = mid + 1
        else:
            high = mid - 1

    return int(best_dim), int(best_params)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_index: int,
    log_every: int,
    logger: logging.Logger,
) -> float:
    """Trains the model for a single epoch.

    Args:
        model: Classifier model that outputs logits or (logits, aux_loss).
        dataloader: Training `DataLoader` yielding (images, labels).
        optimizer: Optimizer instance.
        device: Compute device.
        epoch_index: Zero-based epoch index (for logging only).
        log_every: Log interval in steps.
        logger: Python logger.

    Returns:
        Mean training loss across the epoch as a float.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0.0
    num_batches = 0

    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        if isinstance(outputs, tuple):
            logits, aux_loss = outputs
            loss = loss_fn(logits, targets) + aux_loss
        else:
            logits = outputs
            loss = loss_fn(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        if log_every > 0 and (step % log_every == 0):
            logger.info(
                f"Epoch {epoch_index + 1} step {step:5d} | Loss: {loss.item():.6f}"
            )

    mean_loss = running_loss / max(1, num_batches)
    return mean_loss


@torch.no_grad()
def evaluate(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Evaluates the model on a dataset and computes loss and accuracy.

    Args:
        model: Classifier model that outputs logits or (logits, aux_loss).
        dataloader: Evaluation `DataLoader` yielding (images, labels).
        device: Compute device.

    Returns:
        (mean_loss, accuracy):
            - mean_loss: Average cross-entropy loss across the dataset.
            - accuracy: Top-1 accuracy in [0, 1].
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        if isinstance(outputs, tuple):
            logits, _aux_loss = outputs
        else:
            logits = outputs

        loss = loss_fn(logits, targets)
        preds = logits.argmax(dim=-1)
        correct = (preds == targets).sum().item()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_examples += batch_size

    mean_loss = total_loss / max(1, total_examples)
    accuracy = total_correct / max(1, total_examples)
    return mean_loss, accuracy


def create_summary_table(results: list[dict]) -> Table:
    """Creates a summary table for the training results.

    Args:
        results: A list of per-configuration result dictionaries.

    Returns:
        A `rich.Table` instance ready to be printed.
    """
    table = Table(
        title="MNIST Training Summary",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Dim", justify="right", style="cyan")
    table.add_column("Params", justify="right", style="blue")
    table.add_column("Runtime (s)", justify="right", style="magenta")
    table.add_column("Test Acc", justify="right", style="green")
    table.add_column("Test Loss", justify="right", style="blue")
    table.add_column("Configuration", style="green", max_width=50)
    table.add_column("Status", justify="center")

    for r in results:
        config_str = ", ".join(
            f"{k}={v.__name__ if hasattr(v, '__name__') else v}"
            for k, v in r["config"].items()
            if k
            not in [
                "dim",
                "seq_len",
                "batch_size",
                "epochs",
                "lr",
                "patch_size",
                "model_name",
            ]
        )
        status_style = (
            "bold green"
            if r["status"] == "Success"
            else ("bold yellow" if r["status"] == "Stopped" else "bold red")
        )
        runtime_str = f"{r['runtime']:.2f}" if r["runtime"] >= 0 else "N/A"
        test_acc_str = (
            f"{r['test_accuracy'] * 100:.2f}%" if r["status"] == "Success" else "N/A"
        )
        test_loss_str = f"{r['test_loss']:.6f}" if r["status"] == "Success" else "N/A"

        table.add_row(
            r["config"]["model_name"],
            str(r["config"].get("dim", "?")),
            r["model_size"],
            runtime_str,
            test_acc_str,
            test_loss_str,
            config_str,
            f"[{status_style}]{r['status']}[/]",
        )
    return table


def build_configurations(base_config: dict) -> list[dict]:
    """Produces a list of backbone configurations mirroring verify_training.

    Args:
        base_config: Dict with shared keys, including `dim`, `seq_len`,
            `batch_size`, `epochs`, `lr`, `patch_size`.

    Returns:
        List of configuration dictionaries.
    """
    configurations: list[dict] = [
        # MLP variants
        {"model_name": "mlp", "act_fn": nn.GELU},
        {"model_name": "mlp", "act_fn": nn.ReLU},
        {"model_name": "mlp", "act_fn": nn.SiLU},
        {"model_name": "mlp", "act_fn": ReluSquared},
        {"model_name": "mlp", "act_fn": Gelu2},
        {"model_name": "mlp", "act_fn": BSiLU},
        {"model_name": "mlp", "act_fn": ReluNelu()},
        # MLP parameter variants
        {"model_name": "mlp", "act_fn": nn.GELU, "residual": True},
        {"model_name": "mlp", "act_fn": nn.GELU, "use_norm": False},
        {"model_name": "mlp", "act_fn": nn.GELU, "pre_norm": True},
        # gMLP (baseline + variants)
        {"model_name": "gmlp"},
        {"model_name": "gmlp", "canonical_gate": True},
        {"model_name": "gmlp", "drop_path": 0.1},
        {"model_name": "gmlp", "gate_activation": nn.SiLU},
        {"model_name": "gmlp", "canonical_gate": True, "gate_activation": nn.SiLU},
        {"model_name": "gmlp", "canonical_gate": True, "drop_path": 0.1},
        {"model_name": "gmlp", "canonical_gate": False, "drop_path": 0.1},
        # FeedForward variants (vanilla)
        {"model_name": "feedforward", "glu_variant": "none", "activation": nn.GELU},
        # FeedForward variants (GLU)
        {"model_name": "feedforward", "glu_variant": "glu"},
        {"model_name": "feedforward", "glu_variant": "swiglu"},
        {"model_name": "feedforward", "glu_variant": "geglu"},
        {"model_name": "feedforward", "glu_variant": "reglu"},
        {"model_name": "feedforward", "glu_variant": "bilinear"},
        # FeedForward variants (Masked GLU)
        {"model_name": "feedforward", "glu_variant": "mglu"},
        {"model_name": "feedforward", "glu_variant": "mswiglu"},
        {"model_name": "feedforward", "glu_variant": "mgeglu"},
        {"model_name": "feedforward", "glu_variant": "mreglu"},
        {"model_name": "feedforward", "glu_variant": "mbilinear"},
        # FastFeedForward variants
        {"model_name": "fastfeedforward", "glu_variant": "swiglu"},
        {"model_name": "fastfeedforward", "glu_variant": "geglu"},
        {"model_name": "fastfeedforward", "glu_variant": "mswiglu"},
        # PathWeightedFFF variants
        {"model_name": "pathweightedfff", "depth": 3},
        {"model_name": "pathweightedfff", "depth": 3, "activation": F.silu},
        {"model_name": "pathweightedfff", "depth": 5},
        # nGPT variants
        {"model_name": "ngpt", "scalar_alpha": True},
        {"model_name": "ngpt", "scalar_alpha": False},
        # SwitchFFN variants
        {
            "model_name": "switch_ffn",
            "num_experts": 8,
            "ff_kwargs": {"mult": 2, "glu_variant": "swiglu"},
        },
        {
            "model_name": "switch_ffn",
            "num_experts": 16,
            "ff_kwargs": {"mult": 2, "glu_variant": "geglu"},
        },
    ]

    # Bind base keys to each configuration at call-site.
    return configurations


def run_configuration(
    data_loaders: tuple[DataLoader, DataLoader, DataLoader],
    base_config: dict,
    variant_config: dict,
    device: torch.device,
    logger: logging.Logger,
    early_stop_patience: int,
    early_stop_min_delta: float,
) -> dict:
    """Runs training and evaluation for a single configuration.

    Args:
        data_loaders: Tuple of (train_loader, test_loader) for MNIST.
        base_config: Shared configuration values.
        variant_config: Model-specific overrides.
        device: Compute device.
        logger: Python logger.

    Returns:
        A result dictionary including config, status, metrics, and metadata.
    """
    train_loader, val_loader, test_loader = data_loaders
    current_config = {**base_config, **variant_config}

    # Ensure `seq_len` matches the patch layout for this run
    patch_size = current_config["patch_size"]
    seq_len = (28 // patch_size) ** 2
    current_config["seq_len"] = seq_len

    status = "Success"
    test_loss = -1.0
    test_accuracy = -1.0
    runtime = -1.0
    model_size = "N/A"

    try:
        backbone = get_model(current_config)
        model = MNISTClassifier(
            backbone=backbone, dim=current_config["dim"], patch_size=patch_size
        )
        model_size = get_model_size(model)
        model = model.to(device)

        try:
            model = torch.compile(model)  # type: ignore[assignment]
        except Exception:
            logger.error("Failed to compile model.", exc_info=True)
            raise

        optimizer = torch.optim.Adam(
            model.parameters(), lr=current_config.get("lr", 1e-3)
        )

        start_time = time.time()
        best_val_loss = float("inf")
        best_state: dict | None = None
        patience_counter = 0

        total_epochs = current_config["epochs"]
        for epoch in range(total_epochs):
            train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch_index=epoch,
                log_every=500,
                logger=logger,
            )

            # Evaluate every epoch to support early stopping
            val_loss, val_acc = evaluate(
                model=model, dataloader=val_loader, device=device
            )

            improved = (best_val_loss - val_loss) >= early_stop_min_delta
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best weights in-memory (no disk I/O)
                try:
                    best_state = copy.deepcopy(model.state_dict())
                except Exception:
                    best_state = None
            else:
                patience_counter += 1

            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                logger.info(
                    f"Early stopping at epoch {epoch + 1}/{total_epochs}: "
                    f"val_loss={val_loss:.6f}, best={best_val_loss:.6f}"
                )
                break

        # Load best weights for final test evaluation if available
        if best_state is not None:
            with contextlib.suppress(Exception):
                model.load_state_dict(best_state)
        # Final test set evaluation (test is never used for early stopping)
        test_loss, test_accuracy = evaluate(
            model=model, dataloader=test_loader, device=device
        )
        runtime = time.time() - start_time
    except KeyboardInterrupt:
        status = "Stopped"
        # Best-effort evaluation on current weights if available
        try:
            test_loss, test_accuracy = evaluate(
                model=model, dataloader=test_loader, device=device
            )  # type: ignore[name-defined]
        except Exception:
            test_loss, test_accuracy = -1.0, -1.0
    except Exception as e:  # noqa: BLE001
        status = "FAIL"
        logger.exception(e)

    result = {
        "config": current_config,
        "status": status,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "model_size": model_size,
        "runtime": runtime,
    }
    return result


def main() -> None:
    """Runs MNIST training across model groups and prints a results table.

    CLI arguments:
        --group: Which group of models to run: one of {all, mlps, ff}. Default: all.
        --epochs: Number of training epochs (int). Default: 2.
        --batch-size: Batch size (int). Default: 128.
        --patch-size: Patch size P (int). Must divide 28. Default: 4.
        --num-workers: DataLoader workers (int). Default: 2.
        --val-split: Fraction of training data used for validation (float in (0,1)). Default: 0.1.
        --prefetch-factor: Number of samples prefetched per worker (only if workers>0). Default: 4.

    Shapes documented in function docstrings as per workspace rules.
    """
    parser = argparse.ArgumentParser(description="Train MNIST across model groups")
    parser.add_argument(
        "--group",
        choices=["all", "mlps", "ff"],
        default="all",
        help="Which group of models to run",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Early stop patience in epochs; set 0 to disable.",
    )
    parser.add_argument(
        "--min-delta",
        dest="min_delta",
        type=float,
        default=0.0,
        help="Minimum validation loss improvement to reset patience.",
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="",
        help=(
            "Comma-separated parameter budgets, e.g. '200K,800K'. "
            "When set, each variant is tuned in dim to approximately match each budget."
        ),
    )
    parser.add_argument(
        "--budget-tolerance",
        dest="budget_tolerance",
        type=float,
        default=0.05,
        help="Fractional tolerance for hitting a budget (e.g., 0.05 for ±5%).",
    )
    parser.add_argument(
        "--budget-min-dim",
        dest="budget_min_dim",
        type=int,
        default=4,
        help="Minimum dim considered during budget search.",
    )
    parser.add_argument(
        "--budget-max-dim",
        dest="budget_max_dim",
        type=int,
        default=2048,
        help="Maximum dim considered during budget search.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    logger = logging.getLogger(__name__)

    if 28 % args.patch_size != 0:
        raise SystemExit("--patch-size must evenly divide 28")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data pipeline
    # Inputs: images [B, 1, 28, 28], labels [B]
    from torchvision import (
        datasets,
        transforms,
    )  # Imported here to avoid hard dependency on import-time

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    full_train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Train/val split
    if not (0.0 < args.val_split < 1.0):
        raise SystemExit("--val-split must be in (0, 1)")
    val_size = int(len(full_train_dataset) * args.val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    # Dataloader kwargs with conditional prefetch/persistent tuning
    loader_common_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        # Only valid when using worker processes
        loader_common_kwargs["prefetch_factor"] = max(2, args.prefetch_factor)
        loader_common_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_common_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_common_kwargs,
    )

    base_config = {
        "dim": 128,
        # `seq_len` is set per-run based on patch_size
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": 1e-3,
        "patch_size": args.patch_size,
    }

    configurations = build_configurations(base_config)

    ff_family_names = {
        "feedforward",
        "fastfeedforward",
        "pathweightedfff",
        "switch_ffn",
        "ngpt",
    }
    if args.group == "mlps":
        configurations = [c for c in configurations if "mlp" in c["model_name"]]
    elif args.group == "ff":
        configurations = [
            c for c in configurations if c["model_name"] in ff_family_names
        ]

    results: list[dict] = []
    budgets_list = parse_param_budgets(args.budgets)
    try:
        if budgets_list:
            for variant in configurations:
                for budget in budgets_list:
                    # Search dim to meet budget for this variant
                    chosen_dim, actual_params = estimate_dim_for_budget(
                        base_config=base_config,
                        variant_config=variant,
                        target_params=budget,
                        tolerance=args.budget_tolerance,
                        min_dim=args.budget_min_dim,
                        max_dim=args.budget_max_dim,
                    )
                    budget_str = format_param_count(budget)
                    variant_with_budget = {
                        **variant,
                        "dim": int(chosen_dim),
                        "budget": f"{budget_str}±{int(args.budget_tolerance * 100)}%",
                        "actual_params": format_param_count(actual_params),
                    }
                    logger.info(
                        f"--- Training {variant['model_name']} for budget {budget_str}: "
                        + ", ".join(
                            f"{k}={v.__name__ if hasattr(v, '__name__') else v}"
                            for k, v in variant_with_budget.items()
                            if k not in {"model_name"}
                        )
                    )
                    result = run_configuration(
                        data_loaders=(train_loader, val_loader, test_loader),
                        base_config=base_config,
                        variant_config=variant_with_budget,
                        device=device,
                        logger=logger,
                        early_stop_patience=args.patience,
                        early_stop_min_delta=args.min_delta,
                    )
                    results.append(result)
        else:
            for variant in configurations:
                logger.info(
                    f"--- Training {variant['model_name']} with config variants: "
                    + ", ".join(
                        f"{k}={v.__name__ if hasattr(v, '__name__') else v}"
                        for k, v in variant.items()
                        if k not in {"model_name"}
                    )
                )
                result = run_configuration(
                    data_loaders=(train_loader, val_loader, test_loader),
                    base_config=base_config,
                    variant_config=variant,
                    device=device,
                    logger=logger,
                    early_stop_patience=args.patience,
                    early_stop_min_delta=args.min_delta,
                )
                results.append(result)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Printing partial results...")

    console = Console()
    table = create_summary_table(results)
    console.print(table)


if __name__ == "__main__":
    main()
