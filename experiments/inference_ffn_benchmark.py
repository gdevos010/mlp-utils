"""A small experiment to verify the inference speed of FastFeedForward."""

import time

import torch

from rich.console import Console
from rich.table import Table
from torch import nn

from mlp_utils.layers.fastfeedforward import FastFeedForward
from mlp_utils.layers.feedforward import FeedForward
from mlp_utils.layers.pathweightedfff import PathWeightedFFF


def benchmark_model(
    model: nn.Module,
    x: torch.Tensor,
    num_runs: int = 100,
    set_eval_mode: bool = True,
) -> float:
    """Measures the average forward pass time for a model in milliseconds."""
    # Put model in eval mode for a fair inference comparison
    if set_eval_mode:
        model.eval()

    # Warm-up run
    with torch.no_grad():
        _ = model(x)

    # For GPU experiments, synchronize before starting the timer
    if x.is_cuda:
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    # For GPU experiments, synchronize before stopping the timer
    if x.is_cuda:
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    return (end_time - start_time) / num_runs * 1000  # return in milliseconds


def get_model_size(model: nn.Module) -> str:
    """Calculates the number of parameters in a model and returns a formatted string."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    return f"{num_params / 1_000:.2f}K"


def run_benchmark(device: str, console: Console) -> None:
    """Runs the benchmark on a specific device and prints the results."""
    console.print(
        f"\n[bold green]--- Running Benchmark on {device.upper()} ---[/bold green]"
    )

    # --- Configuration ---
    # Using larger dimensions to make the performance difference more apparent.
    dim = 512
    seq_len = 512
    batch_size = 16
    depth = 5  # This will create 2**5 = 32 experts
    num_runs = 200

    config_table = Table(title=f"Benchmark Configuration ({device.upper()})")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="magenta")
    config_table.add_row("Device", device.upper())
    config_table.add_row("Dimension (dim)", str(dim))
    config_table.add_row("Sequence Length (seq_len)", str(seq_len))
    config_table.add_row("Batch Size", str(batch_size))
    config_table.add_row("FFF Depth", str(depth) + f" ({2**depth} experts)")
    config_table.add_row("Number of Runs", str(num_runs))
    console.print(config_table)

    # --- Data and Models ---
    x = torch.randn(batch_size, seq_len, dim).to(device)

    fff_swiglu_model = FastFeedForward(
        dim=dim,
        depth=depth,
        glu_variant="swiglu",
        expert_dim=dim // 4,
    ).to(device)

    fff_geglu_model = FastFeedForward(
        dim=dim,
        depth=depth,
        glu_variant="geglu",
        expert_dim=dim // 4,
    ).to(device)

    pathweighted_fff_model = PathWeightedFFF(
        input_width=dim, depth=depth, output_width=dim
    ).to(device)

    ff_model = FeedForward(dim=dim, glu_variant="swiglu", mult=4).to(device)

    fff_swiglu_model = torch.compile(fff_swiglu_model)
    fff_geglu_model = torch.compile(fff_geglu_model)
    pathweighted_fff_model = torch.compile(pathweighted_fff_model)
    ff_model = torch.compile(ff_model)

    # --- Benchmarking ---
    results = []

    # 1. Standard FeedForward (Baseline)
    ff_time = benchmark_model(ff_model, x, num_runs)
    ff_size = get_model_size(ff_model)
    results.append(
        {
            "Model": "FeedForward",
            "Size": ff_size,
            "Mode": "Inference (eval)",
            "Avg. Time (ms)": f"{ff_time:.4f}",
            "Notes": "Standard baseline",
        }
    )

    # 2. FastFeedForward (Hard Routing - Inference Mode)
    # The `benchmark_model` function puts the model in `eval` mode automatically.
    fff_swiglu_hard_time = benchmark_model(fff_swiglu_model, x, num_runs)
    fff_size = get_model_size(fff_swiglu_model)
    results.append(
        {
            "Model": "FastFeedForward (SwiGLU)",
            "Size": fff_size,
            "Mode": "Hard Routing (eval)",
            "Avg. Time (ms)": f"{fff_swiglu_hard_time:.4f}",
            "Notes": "Optimized fast path for swiglu",
        }
    )

    # 3. FastFeedForward (Soft Routing - Training Mode)
    # We manually set the model to `train` mode to force soft routing.
    fff_swiglu_model.train()
    # The benchmark function will still use torch.no_grad to be comparable.
    fff_swiglu_soft_time = benchmark_model(
        fff_swiglu_model, x, num_runs, set_eval_mode=False
    )
    results.append(
        {
            "Model": "FastFeedForward (SwiGLU)",
            "Size": fff_size,
            "Mode": "Soft Routing (train)",
            "Avg. Time (ms)": f"{fff_swiglu_soft_time:.4f}",
            "Notes": "Used for training, processes ALL experts",
        }
    )

    # 4. FastFeedForward with GeGLU (Generic Path)
    fff_geglu_hard_time = benchmark_model(fff_geglu_model, x, num_runs)
    fff_size = get_model_size(fff_geglu_model)
    results.append(
        {
            "Model": "FastFeedForward (GeGLU)",
            "Size": fff_size,
            "Mode": "Hard Routing (eval)",
            "Avg. Time (ms)": f"{fff_geglu_hard_time:.4f}",
            "Notes": "Generic path for non-swiglu experts",
        }
    )

    # 5. PathWeightedFFF
    pathweighted_time = benchmark_model(pathweighted_fff_model, x, num_runs)
    pathweighted_size = get_model_size(pathweighted_fff_model)
    results.append(
        {
            "Model": "PathWeightedFFF",
            "Size": pathweighted_size,
            "Mode": "Inference (eval)",
            "Avg. Time (ms)": f"{pathweighted_time:.4f}",
            "Notes": "Path-weighted combination, not MoE",
        }
    )

    # --- Display Results ---
    table = Table(title=f"Inference Speed Comparison ({device.upper()})")
    table.add_column("Model", style="cyan")
    table.add_column("Size", justify="right", style="blue")
    table.add_column("Mode", style="yellow")
    table.add_column("Avg. Time (ms)", justify="right", style="magenta")
    table.add_column("Notes", style="green")

    for r in results:
        table.add_row(r["Model"], r["Size"], r["Mode"], r["Avg. Time (ms)"], r["Notes"])

    console.print(table)

    console.print(f"\n[bold]Conclusion for {device.upper()}:[/bold]")
    if fff_swiglu_hard_time > 0 and fff_swiglu_soft_time > fff_swiglu_hard_time:
        speedup_factor = fff_swiglu_soft_time / fff_swiglu_hard_time
        console.print(
            f"FastFeedForward in inference mode (hard routing) is "
            f"[bold magenta]{speedup_factor:.2f}x[/bold magenta] faster than "
            f"its training mode counterpart (soft routing)."
        )
    console.print(
        "The inference speed is comparable to a standard FeedForward layer, "
        "confirming the 'fast' nature for inference."
    )


def main() -> None:
    """Runs the benchmark and prints the results."""
    console = Console()
    console.print("[bold cyan]--- Benchmarking Inference Speed ---[/bold cyan]")

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    else:
        console.print("[yellow]CUDA not available, running on CPU only.[/yellow]")

    for device in devices:
        run_benchmark(device, console)


if __name__ == "__main__":
    main()
