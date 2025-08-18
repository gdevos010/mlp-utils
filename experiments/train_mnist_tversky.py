# ABOUTME: Trains MNIST using a TverskyProjection classifier head.
# ABOUTME: Demonstrates prototype-based classification with configurable reductions.

"""MNIST demo using `TverskyProjection` as the classifier.

This script builds a simple sequence-based encoder for MNIST images and replaces
the final linear classifier with a Tversky similarity projection. It supports
multiple prototypes per class with optional pooling across prototypes.

Key hyperparameters are exposed to experiment with asymmetric similarity
(`alpha`, `beta`), stability (`theta`), smoothing, prototype initialization,
and reduction strategies.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import torch

from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from mlp_utils.layers import TverskyProjection
from mlp_utils.layers.mlp import MLP

IMAGE_NDIM = 4
IMAGE_CHANNELS = 1
IMAGE_SIZE = 28

logger = logging.getLogger(__name__)

MNIST_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(MNIST_DATA_DIR, exist_ok=True)


def patchify_images(images: Tensor, patch_size: int) -> Tensor:
    """Converts images [B, 1, 28, 28] into sequences of flattened patches [B, S, P^2]."""
    if (
        images.ndim != IMAGE_NDIM
        or images.shape[1] != IMAGE_CHANNELS
        or images.shape[2] != IMAGE_SIZE
        or images.shape[3] != IMAGE_SIZE
    ):
        raise ValueError(
            f"Expected images with shape [B, 1, 28, 28], got {tuple(images.shape)}"
        )
    if IMAGE_SIZE % patch_size != 0:
        raise ValueError("patch_size must evenly divide 28")

    batch_size = images.shape[0]
    patches = images.unfold(dimension=2, size=patch_size, step=patch_size).unfold(
        dimension=3, size=patch_size, step=patch_size
    )
    patches = patches.permute(0, 2, 3, 1, 4, 5)  # [B, H, W, 1, P, P]
    patches = patches.reshape(
        batch_size,
        (IMAGE_SIZE // patch_size) * (IMAGE_SIZE // patch_size),
        patch_size * patch_size,
    )
    return patches


class MNISTTverskyClassifier(nn.Module):
    """Classifier using a backbone encoder and TverskyProjection head.

    Shapes:
        - Input images: [B, 1, 28, 28]
        - After patchify: [B, S, P^2], S = (28 / P)^2
        - After input projection: [B, S, D]
        - Backbone output: [B, S, D]
        - Pooled features: [B, D]
        - Head output: [B, C] (after prototype pooling if P>1)
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        backbone: nn.Module,
        dim: int,
        patch_size: int,
        num_classes: int = 10,
        num_prototypes_per_class: int = 1,
        prototype_pool: str = "mean",  # {"mean", "max"}
        # TverskyProjection kwargs
        alpha: float = 0.5,
        beta: float = 0.5,
        theta: float = 1e-7,
        input_transform: str | None = "relu",
        nonnegative: bool = True,
        smoothing_tau: float | None = None,
        prototype_init: str = "xavier",
        intersection_reduction: str = "sum",
        difference_reduction: str = "subtractmatch",
        match_threshold: float = 0.0,
        temperature: float | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if prototype_pool not in {"mean", "max"}:
            raise ValueError("prototype_pool must be one of 'mean' or 'max'")
        self.backbone = backbone
        self.input_proj = nn.Linear(patch_size * patch_size, dim)
        self.patch_size = patch_size
        self.num_classes = int(num_classes)
        self.num_prototypes_per_class = int(num_prototypes_per_class)
        self.prototype_pool = prototype_pool

        total_prototypes = int(num_classes) * int(num_prototypes_per_class)
        self.tversky_head = TverskyProjection(
            input_dim=dim,
            output_dim=total_prototypes,
            alpha=alpha,
            beta=beta,
            theta=theta,
            bias=bias,
            input_transform=input_transform,
            nonnegative=nonnegative,
            smoothing_tau=smoothing_tau,
            prototype_init=prototype_init,
            intersection_reduction=intersection_reduction,
            difference_reduction=difference_reduction,
            match_threshold=match_threshold,
            temperature=temperature,
        )

    def forward(self, images: Tensor) -> Tensor:
        """Compute class logits from input images.

        Args:
            images: Tensor of shape [B, 1, 28, 28].

        Returns:
            Logits tensor of shape [B, C].
        """
        patches = patchify_images(images, self.patch_size)  # [B, S, P^2]
        tokens = self.input_proj(patches)  # [B, S, D]
        features = self.backbone(tokens)  # expected [B, S, D]
        if isinstance(features, tuple):
            features = features[0]
        pooled = features.mean(dim=1)  # [B, D]
        sims = self.tversky_head(pooled)  # [B, C * P]
        if self.num_prototypes_per_class > 1:
            b = sims.shape[0]
            sims = sims.view(
                b, self.num_classes, self.num_prototypes_per_class
            )  # [B, C, P]
            if self.prototype_pool == "mean":
                sims = sims.mean(dim=-1)
            else:  # max
                sims = sims.max(dim=-1).values
        else:
            sims = sims.view(sims.shape[0], self.num_classes)
        return sims


def build_backbone(dim: int, seq_len: int) -> nn.Module:
    """Returns a simple MLP backbone that maps [B, S, D] -> [B, S, D]."""
    return MLP(
        input_dim=dim,
        output_dim=dim,
        hidden_factor=4,
        act_fn=nn.GELU,
        residual=True,
        use_norm=True,
        norm_mode="post",
    )


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Evaluates model on a loader, returns (mean_loss, accuracy)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images_batch, targets_batch in loader:
        images_dev = images_batch.to(device, non_blocking=True)
        targets_dev = targets_batch.to(device, non_blocking=True)
        logits = model(images_dev)
        loss = loss_fn(logits, targets_dev)
        preds = logits.argmax(dim=-1)
        correct = (preds == targets_dev).sum().item()
        bs = images_dev.shape[0]
        total_loss += loss.item() * bs
        total_correct += correct
        total_examples += bs
    return total_loss / max(1, total_examples), total_correct / max(1, total_examples)


@torch.no_grad()
def seed_prototypes_from_dataset(
    model: MNISTTverskyClassifier,
    dataset: torch.utils.data.Dataset,
    *,
    device: torch.device,
    samples_per_class: int,
) -> None:
    """Seed prototypes using pooled features from dataset examples per class.

    For each class c, collects up to `samples_per_class` examples, computes
    pooled features, and assigns them to the corresponding class prototype
    slots in order. If fewer examples are available than prototypes, features
    are repeated cyclically.
    """
    model.eval()
    num_classes = model.num_classes
    protos_per_class = model.num_prototypes_per_class
    per_class_feats: list[list[Tensor]] = [[] for _ in range(num_classes)]

    # Create a small loader to batch feature extraction
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    for images_batch, targets_batch in loader:
        # Stop early if all classes are filled
        if all(
            len(per_class_feats[c]) >= samples_per_class for c in range(num_classes)
        ):
            break
        images_dev = images_batch.to(device)
        tokens = model.input_proj(patchify_images(images_dev, model.patch_size))
        features = model.backbone(tokens)
        if isinstance(features, tuple):
            features = features[0]
        pooled = features.mean(dim=1)  # [B, D]
        for vec, cls in zip(pooled, targets_batch, strict=False):
            c = int(cls.item())
            if len(per_class_feats[c]) < samples_per_class:
                per_class_feats[c].append(vec.detach().cpu())

    # Assign into prototype weights
    weight = model.tversky_head.weight  # [C*P, D]
    feat_dim = weight.shape[1]
    for c in range(num_classes):
        feats = per_class_feats[c]
        if not feats:
            # Fallback to zeros if no samples collected (should not happen)
            feats = [torch.zeros(feat_dim)]
        for k in range(protos_per_class):
            idx = c * protos_per_class + k
            src = feats[k % len(feats)].to(weight.dtype)
            weight[idx].copy_(src[:feat_dim])


def train(  # noqa: PLR0913
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    patch_size: int,
    dim: int,
    num_prototypes_per_class: int,
    prototype_pool: str,
    alpha: float,
    beta: float,
    theta: float,
    input_transform: str | None,
    nonnegative: bool,
    smoothing_tau: float | None,
    prototype_init: str,
    intersection_reduction: str,
    difference_reduction: str,
    match_threshold: float,
    temperature: float | None,
    bias: bool,
    seed_prototypes: bool,
    seed_samples_per_class: int,
) -> None:
    """Trains the MNIST Tversky demo and logs validation/test metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_full = datasets.MNIST(
        root=MNIST_DATA_DIR, train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root=MNIST_DATA_DIR, train=False, download=True, transform=transform
    )

    val_split = 0.1
    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(train_full, [train_size, val_size])

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 2,
        "pin_memory": torch.cuda.is_available(),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    seq_len = (IMAGE_SIZE // patch_size) ** 2
    backbone = build_backbone(dim=dim, seq_len=seq_len)
    model = MNISTTverskyClassifier(
        backbone=backbone,
        dim=dim,
        patch_size=patch_size,
        num_classes=10,
        num_prototypes_per_class=num_prototypes_per_class,
        prototype_pool=prototype_pool,
        alpha=alpha,
        beta=beta,
        theta=theta,
        input_transform=input_transform,
        nonnegative=nonnegative,
        smoothing_tau=smoothing_tau,
        prototype_init=prototype_init,
        intersection_reduction=intersection_reduction,
        difference_reduction=difference_reduction,
        match_threshold=match_threshold,
        temperature=temperature,
        bias=bias,
    ).to(device)

    if seed_prototypes:
        seed_prototypes_from_dataset(
            model=model,
            dataset=train_full,
            device=device,
            samples_per_class=max(1, seed_samples_per_class),
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    logger.info(
        f"Training on {device} | epochs={epochs}, batch_size={batch_size}, lr={lr}, dim={dim}, P={patch_size}, "
        f"protos/class={num_prototypes_per_class}, pool={prototype_pool}"
    )

    best_val = float("inf")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images_batch, targets_batch in train_loader:
            images_dev = images_batch.to(device, non_blocking=True)
            targets_dev = targets_batch.to(device, non_blocking=True)
            logits = model(images_dev)
            loss = loss_fn(logits, targets_dev)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= max(1, len(train_loader))
        val_loss, val_acc = evaluate(model, val_loader, device)
        logger.info(
            f"Epoch {epoch:03d} | train_loss={epoch_loss:.6f} | val_loss={val_loss:.6f} | val_acc={val_acc * 100:.2f}%"
        )
        best_val = min(best_val, val_loss)

    elapsed = time.time() - start_time
    test_loss, test_acc = evaluate(model, test_loader, device)
    logger.info(
        f"Done in {elapsed:.2f}s | best_val_loss={best_val:.6f} | test_loss={test_loss:.6f} | test_acc={test_acc * 100:.2f}%"
    )


def perform_ray_tuning(args: argparse.Namespace) -> None:  # noqa: PLR0915
    """Runs Ray Tune hyperparameter search for the MNIST Tversky demo.

    Searches across a small space (lr, prototypes per class, pooling, transforms,
    reductions, and logit scaling). Reports validation accuracy per epoch and
    logs the best configuration summary.
    """
    try:
        from ray import tune  # noqa: PLC0415
        from ray.tune.schedulers import ASHAScheduler  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Ray is not installed or failed to import. Install with `pip install ray[tune]`."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def trainable(config: dict) -> None:  # noqa: C901, PLR0915
        # Resolve config values with CLI defaults
        epochs = int(config.get("epochs", args.epochs))
        batch_size = int(config.get("batch_size", args.batch_size))
        patch_size = int(config.get("patch_size", args.patch_size))
        dim = int(config.get("dim", args.dim))
        num_prototypes_per_class = int(
            config.get("prototypes_per_class", args.prototypes_per_class)
        )
        prototype_pool = config.get("prototype_pool", args.prototype_pool)
        lr = float(config.get("lr", args.lr))
        alpha = float(config.get("alpha", args.alpha))
        beta = float(config.get("beta", args.beta))
        theta = float(config.get("theta", args.theta))
        input_transform = config.get("input_transform", args.input_transform)
        if input_transform == "none":
            input_transform = None
        nonnegative = bool(config.get("nonnegative", args.nonnegative))
        smoothing = config.get("smoothing_tau", args.smoothing_tau)
        smoothing_tau = (
            None if (smoothing is None or float(smoothing) <= 0.0) else float(smoothing)
        )
        prototype_init = config.get("prototype_init", args.prototype_init)
        intersection_reduction = config.get(
            "intersection_reduction", args.intersection_reduction
        )
        difference_reduction = config.get(
            "difference_reduction", args.difference_reduction
        )
        match_threshold = float(config.get("match_threshold", args.match_threshold))
        temp_val = config.get("temperature", args.temperature)
        temperature = (
            None if (temp_val is None or float(temp_val) <= 0.0) else float(temp_val)
        )
        bias = bool(config.get("bias", args.bias))
        seed_protos = bool(config.get("seed_prototypes", args.seed_prototypes))
        seed_k = int(config.get("seed_samples_per_class", args.seed_samples_per_class))

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_full = datasets.MNIST(
            root=MNIST_DATA_DIR, train=True, download=True, transform=transform
        )
        test_set = datasets.MNIST(
            root=MNIST_DATA_DIR, train=False, download=True, transform=transform
        )

        val_split = 0.1
        val_size = int(len(train_full) * val_split)
        train_size = len(train_full) - val_size
        train_set, val_set = random_split(train_full, [train_size, val_size])

        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": 2,
            "pin_memory": torch.cuda.is_available(),
        }
        if loader_kwargs["num_workers"] > 0:
            loader_kwargs["prefetch_factor"] = 4
            loader_kwargs["persistent_workers"] = True

        train_loader = DataLoader(
            train_set, shuffle=True, drop_last=True, **loader_kwargs
        )
        val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

        seq_len = (IMAGE_SIZE // patch_size) ** 2
        backbone = build_backbone(dim=dim, seq_len=seq_len)
        model = MNISTTverskyClassifier(
            backbone=backbone,
            dim=dim,
            patch_size=patch_size,
            num_classes=10,
            num_prototypes_per_class=num_prototypes_per_class,
            prototype_pool=prototype_pool,
            alpha=alpha,
            beta=beta,
            theta=theta,
            input_transform=input_transform,
            nonnegative=nonnegative,
            smoothing_tau=smoothing_tau,
            prototype_init=prototype_init,
            intersection_reduction=intersection_reduction,
            difference_reduction=difference_reduction,
            match_threshold=match_threshold,
            temperature=temperature,
            bias=bias,
        ).to(device)

        if seed_protos:
            seed_prototypes_from_dataset(
                model=model,
                dataset=train_full,
                device=device,
                samples_per_class=max(1, seed_k),
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            model.train()
            for images_batch, targets_batch in train_loader:
                images_dev = images_batch.to(device, non_blocking=True)
                targets_dev = targets_batch.to(device, non_blocking=True)
                logits = model(images_dev)
                loss = loss_fn(logits, targets_dev)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            val_loss, val_acc = evaluate(model, val_loader, device)
            tune.report(
                {"val_loss": float(val_loss), "val_acc": float(val_acc), "epoch": epoch}
            )

        test_loss, test_acc = evaluate(model, test_loader, device)
        # Ensure the scheduler's metric (val_acc) is present in the final report
        val_loss_final, val_acc_final = evaluate(model, val_loader, device)
        epoch_final = int(epochs)
        tune.report(
            {
                "epoch": epoch_final,
                "val_loss": float(val_loss_final),
                "val_acc": float(val_acc_final),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
            }
        )

        # Note: returning a dict here is unnecessary; Tune consumes metrics via tune.report

    # Define search space
    search_space = {
        "lr": tune.loguniform(1e-3, 3e-2),
        "prototypes_per_class": tune.choice([1, 2, 4]),
        "prototype_pool": tune.choice(["mean", "max"]),
        "alpha": tune.choice([0.3, 0.5, 0.7]),
        "beta": tune.choice([0.3, 0.5, 0.7]),
        "intersection_reduction": tune.choice(["sum", "mean"]),
        "input_transform": tune.choice(["relu", "clamp01", "sigmoid", "none"]),
        "smoothing_tau": tune.choice([0.0, 0.5]),
        "bias": tune.choice([False, True]),
        "temperature": tune.choice([0.0, 5.0, 10.0]),
        "seed_prototypes": tune.choice([False, True]),
        "seed_samples_per_class": tune.choice([2, 4, 8]),
        # Fixed from CLI to keep search bounded
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "dim": args.dim,
        "theta": args.theta,
        "difference_reduction": args.difference_reduction,
        "match_threshold": args.match_threshold,
        "prototype_init": args.prototype_init,
    }

    scheduler = ASHAScheduler(
        time_attr="epoch",
        metric="val_acc",
        mode="max",
        grace_period=3,
        reduction_factor=2,
    )

    resources = {"cpu": 4, "gpu": 1 if torch.cuda.is_available() else 0}

    storage_uri = f"file://{os.path.abspath('./ray_results')}"
    analysis = tune.run(
        trainable,
        name="mnist_tversky_tune",
        config=search_space,
        scheduler=scheduler,
        num_samples=int(args.tune_samples),
        resources_per_trial=resources,
        storage_path=storage_uri,
    )

    best_config = analysis.get_best_config(metric="val_acc", mode="max")
    best_result = analysis.get_best_trial(metric="val_acc", mode="max", scope="all")
    logger.info(f"Best config: {best_config}")
    logger.info(f"Best result: {best_result.metric_analysis.get('val_acc')}")


def main() -> None:
    """CLI entrypoint for the MNIST TverskyProjection demo."""
    parser = argparse.ArgumentParser(description="MNIST TverskyProjection demo")
    # Data/model
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--prototypes-per-class", type=int, default=1)
    parser.add_argument("--prototype-pool", choices=["mean", "max"], default="mean")
    # Tversky params
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--theta", type=float, default=1e-7)
    parser.add_argument("--input-transform", type=str, default="relu")
    parser.add_argument("--nonnegative", action="store_true", default=True)
    parser.add_argument("--no-nonnegative", dest="nonnegative", action="store_false")
    parser.add_argument("--smoothing-tau", type=float, default=None)
    parser.add_argument(
        "--prototype-init", choices=["xavier", "kaiming"], default="xavier"
    )
    parser.add_argument(
        "--intersection-reduction",
        choices=["sum", "mean", "product"],
        default="sum",
    )
    parser.add_argument(
        "--difference-reduction",
        choices=["subtractmatch", "ignorematch"],
        default="subtractmatch",
    )
    parser.add_argument("--match-threshold", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--bias", action="store_true", default=False)
    parser.add_argument("--seed-prototypes", action="store_true", default=False)
    parser.add_argument("--seed-samples-per-class", type=int, default=4)
    # Ray Tune options
    parser.add_argument("--tune", action="store_true", default=False)
    parser.add_argument("--tune-samples", type=int, default=20)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.tune:
        perform_ray_tuning(args)
    else:
        train(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            dim=args.dim,
            num_prototypes_per_class=args.prototypes_per_class,
            prototype_pool=args.prototype_pool,
            alpha=args.alpha,
            beta=args.beta,
            theta=args.theta,
            input_transform=args.input_transform
            if args.input_transform != "none"
            else None,
            nonnegative=args.nonnegative,
            smoothing_tau=args.smoothing_tau,
            prototype_init=args.prototype_init,
            intersection_reduction=args.intersection_reduction,
            difference_reduction=args.difference_reduction,
            match_threshold=args.match_threshold,
            temperature=args.temperature,
            bias=args.bias,
            seed_prototypes=args.seed_prototypes,
            seed_samples_per_class=args.seed_samples_per_class,
        )


if __name__ == "__main__":
    main()
