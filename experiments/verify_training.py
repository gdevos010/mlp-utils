import logging
import time

import torch
import torch.nn.functional as F

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from torch import nn

from mlp_utils.layers.fastfeedforward import FastFeedForward
from mlp_utils.layers.feedforward import FeedForward
from mlp_utils.layers.gmlp import GMLP
from mlp_utils.layers.mlp import MLP
from mlp_utils.layers.ngpt import NGPT
from mlp_utils.layers.pathweightedfff import PathWeightedFFF
from mlp_utils.layers.switch_ffn import SwitchFFN


def get_synthetic_data(batch_size, seq_len, dim):
    """Generates a batch of synthetic data for a regression task."""
    x = torch.randn(batch_size, seq_len, dim)
    # A simple, non-linear function for the models to learn
    y_true = torch.sin(x) + torch.cos(x * 2) * 0.5 + torch.randn_like(x) * 0.01
    return x, y_true


def get_model(config: dict) -> nn.Module:
    """Instantiates a model based on the provided configuration."""
    model_name = config["model_name"]
    dim = config["dim"]
    seq_len = config["seq_len"]

    if model_name == "mlp":
        return MLP(
            input_dim=dim,
            output_dim=dim,
            hidden_factor=4,
            act_fn=config["act_fn"],
            residual=config.get("residual", False),
            use_norm=config.get("use_norm", True),
            pre_norm=config.get("pre_norm", False),
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
            dim=dim, depth=3, mult=4, glu_variant=config["glu_variant"]
        )
    if model_name == "pathweightedfff":
        return PathWeightedFFF(input_width=dim, depth=config["depth"], output_width=dim)
    if model_name == "ngpt":
        ff_net = FeedForward(dim=dim, mult=4, glu_variant="swiglu")
        return NGPT(
            feedforward_net=ff_net, dim=dim, scalar_alpha=config["scalar_alpha"]
        )
    if model_name == "gmlp":
        return GMLP(dim=dim, dim_ff=dim * 4, seq_len=seq_len, depth=4)
    if model_name == "switch_ffn":
        return SwitchFFN(
            dim=dim,
            num_experts=config["num_experts"],
            ff_kwargs=config["ff_kwargs"],
        )
    raise ValueError(f"Unknown model: {model_name}")


def get_model_size(model: nn.Module) -> str:
    """Calculates the number of parameters in a model."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_params >= 1e6:  # noqa: PLR2004
        return f"{num_params / 1e6:.2f}M"
    if num_params >= 1e3:  # noqa: PLR2004
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def train(model: nn.Module, model_config: dict, use_compile: bool) -> float:
    """Trains the given model on the synthetic task."""
    logger = logging.getLogger(__name__)
    lr = model_config.get("lr", 1e-3)
    steps = model_config.get("steps", 201)
    batch_size = model_config.get("batch_size", 16)
    dim = model_config["dim"]
    seq_len = model_config["seq_len"]

    if use_compile:
        try:
            model = torch.compile(model)
        except Exception:
            logger.error("Failed to compile model.", exc_info=True)
            raise

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    config_str = ", ".join(
        f"{k}={v.__name__ if hasattr(v, '__name__') else v}"
        for k, v in model_config.items()
        if k not in ["dim", "seq_len", "batch_size", "steps", "lr"]
    )
    logger.info(
        f"--- Training {model_config['model_name']} (compile={use_compile}) with config: {config_str} ---"
    )

    for step in range(steps):
        x, y_true = get_synthetic_data(batch_size, seq_len, dim)
        y_pred = model(x)

        # For models that operate on the hypersphere, the target must also be normalized
        if getattr(model, "needs_normalized_target", False):
            y_true = F.normalize(y_true, p=2, dim=-1)

        # Handle models with auxiliary loss (e.g., SwitchFFN)
        if getattr(model, "has_aux_loss", False):
            y_pred, aux_loss = y_pred
            loss = loss_fn(y_pred, y_true) + aux_loss
        else:
            loss = loss_fn(y_pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            logger.info(f"Step {step:4d}/{steps}, Loss: {loss.item():.6f}")

    logger.info("--- Training finished for config ---")
    return loss.item()


def create_summary_table(results: list[dict]) -> Table:
    """Creates a rich Table for the training results."""
    table = Table(
        title="Training Verification Summary",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Compile", style="yellow")
    table.add_column("Params", justify="right", style="blue")
    table.add_column("Runtime (s)", justify="right", style="magenta")
    table.add_column("Configuration", style="green", max_width=50)
    table.add_column("Final Loss", justify="right", style="blue")
    table.add_column("Status", justify="center")

    for r in results:
        config_str = ", ".join(
            f"{k}={v.__name__ if hasattr(v, '__name__') else v}"
            for k, v in r["config"].items()
            if k not in ["dim", "seq_len", "batch_size", "steps", "lr", "model_name"]
        )
        final_loss_str = f"{r['final_loss']:.6f}" if r["status"] == "Success" else "N/A"
        status_style = "bold green" if r["status"] == "Success" else "bold red"
        runtime_str = f"{r['runtime']:.2f}" if r["runtime"] >= 0 else "N/A"

        table.add_row(
            r["config"]["model_name"],
            str(r["compiled"]),
            r["model_size"],
            runtime_str,
            config_str,
            final_loss_str,
            f"[{status_style}]{r['status']}[/]",
        )
    return table


def main() -> None:
    """Runs the training experiment for a variety of model configurations."""
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    logger = logging.getLogger(__name__)

    base_config = {
        "dim": 64,
        "seq_len": 64,
        "batch_size": 32,
        "steps": 501,
        "lr": 1e-3,
    }

    configurations = [
        # MLP variants
        # {"model_name": "mlp", "act_fn": nn.GELU},
        # {"model_name": "mlp", "act_fn": nn.ReLU},
        # {"model_name": "mlp", "act_fn": nn.SiLU},
        # {"model_name": "mlp", "act_fn": ReluSquared},
        # {"model_name": "mlp", "act_fn": Gelu2},
        # {"model_name": "mlp", "act_fn": BSiLU},
        # {"model_name": "mlp", "act_fn": ReluNelu()},
        # # MLP parameter variants
        # {"model_name": "mlp", "act_fn": nn.GELU, "residual": True},
        # {"model_name": "mlp", "act_fn": nn.GELU, "use_norm": False},
        # {"model_name": "mlp", "act_fn": nn.GELU, "pre_norm": True},
        # # FeedForward variants (vanilla)
        # {"model_name": "feedforward", "glu_variant": "none", "activation": nn.GELU},
        # # FeedForward variants (GLU)
        # {"model_name": "feedforward", "glu_variant": "glu"},
        # {"model_name": "feedforward", "glu_variant": "swiglu"},
        # {"model_name": "feedforward", "glu_variant": "geglu"},
        # {"model_name": "feedforward", "glu_variant": "reglu"},
        # {"model_name": "feedforward", "glu_variant": "bilinear"},
        # # FeedForward variants (Masked GLU)
        # {"model_name": "feedforward", "glu_variant": "mglu"},
        # {"model_name": "feedforward", "glu_variant": "mswiglu"},
        # {"model_name": "feedforward", "glu_variant": "mgeglu"},
        # {"model_name": "feedforward", "glu_variant": "mreglu"},
        # {"model_name": "feedforward", "glu_variant": "mbilinear"},
        # # FastFeedForward variants
        # {
        #     "model_name": "fastfeedforward",
        #     "glu_variant": "swiglu",
        #     "expert_dim": base_config["dim"] // 8,
        # },
        # {
        #     "model_name": "fastfeedforward",
        #     "glu_variant": "geglu",
        #     "expert_dim": base_config["dim"] // 8,
        # },
        # {
        #     "model_name": "fastfeedforward",
        #     "glu_variant": "mswiglu",
        #     "expert_dim": base_config["dim"] // 8,
        # },
        # # FastFeedForward with load balancing and master leaf
        # {
        #     "model_name": "fastfeedforward",
        #     "glu_variant": "swiglu",
        #     "expert_dim": base_config["dim"] // 8,
        # },
        # {
        #     "model_name": "fastfeedforward",
        #     "glu_variant": "swiglu",
        #     "expert_dim": base_config["dim"] // 8,
        # },
        # {
        #     "model_name": "fastfeedforward",
        #     "glu_variant": "swiglu",
        #     "expert_dim": base_config["dim"] // 8,
        # },
        # PathWeightedFFF variants
        {"model_name": "pathweightedfff", "depth": 3},
        {"model_name": "pathweightedfff", "depth": 5},
        # nGPT variants
        {"model_name": "ngpt", "scalar_alpha": True},
        {"model_name": "ngpt", "scalar_alpha": False},
        # gMLP
        {"model_name": "gmlp"},
        # SwitchFFN variants
        {
            "model_name": "switch_ffn",
            "num_experts": 8,
            "ff_kwargs": {"mult": 4, "glu_variant": "swiglu"},
        },
        {
            "model_name": "switch_ffn",
            "num_experts": 16,
            "ff_kwargs": {"mult": 2, "glu_variant": "geglu"},
        },
    ]

    results = []
    compile_mode = True
    for config in configurations:
        current_config = {**base_config, **config}
        status = "Success"
        final_loss = -1.0
        model_size = "N/A"
        runtime = -1.0
        try:
            model = get_model(current_config)
            model_size = get_model_size(model)
            start_time = time.time()
            final_loss = train(model, current_config, use_compile=compile_mode)
            runtime = time.time() - start_time
        except Exception as e:
            status = "FAIL"

            logger.exception(e)

        results.append(
            {
                "config": current_config,
                "compiled": compile_mode,
                "status": status,
                "final_loss": final_loss,
                "model_size": model_size,
                "runtime": runtime,
            }
        )

    console = Console()
    summary_table = create_summary_table(results)
    console.print(summary_table)


if __name__ == "__main__":
    main()
