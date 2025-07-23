"""Profiling script for GLU variants in mlp_utils.

This script benchmarks the performance of different GLU implementations,
including both standard GLU and Masked GLU (MGLU) variants.


"""

import time
import tracemalloc

import torch

from tabulate import tabulate
from torch import nn

from mlp_utils.layers.glu import (
    GLU,
    MGLU,
    Bilinear,
    BilinearMGLU,
    GeGLU,
    GeMGLU,
    ReGLU,
    ReMGLU,
    SwiGLU,
    SwiMGLU,
)


class GLUBenchmark:
    """Benchmark suite for GLU variants."""

    def __init__(
        self, device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        self.device = device

        # GLU variants to benchmark
        self.glu_variants = {
            "GLU": GLU,
            "Bilinear": Bilinear,
            "ReGLU": ReGLU,
            "SwiGLU": SwiGLU,
            "GeGLU": GeGLU,
        }

        self.mglu_variants = {
            "MGLU": MGLU,
            "BilinearMGLU": BilinearMGLU,
            "ReMGLU": ReMGLU,
            "SwiMGLU": SwiMGLU,
            "GeMGLU": GeMGLU,
        }

        # Test configurations
        self.test_configs = [
            {"batch_size": 32, "dim_in": 256, "dim_out": 1024},
            {"batch_size": 64, "dim_in": 512, "dim_out": 2048},
            {"batch_size": 128, "dim_in": 768, "dim_out": 3072},
            {"batch_size": 256, "dim_in": 1024, "dim_out": 4096},
        ]

        self.warmup_runs = 10
        self.benchmark_runs = 500

    def create_test_data(self, batch_size: int, dim_in: int) -> torch.Tensor:
        """Create test input tensor."""
        return torch.randn(batch_size, dim_in, device=self.device, requires_grad=True)

    def measure_forward_time(self, model: nn.Module, x: torch.Tensor) -> float:
        """Measure forward pass time."""
        # Warmup
        for _ in range(self.warmup_runs):
            _ = model(x)
            if self.device == "cuda":
                torch.cuda.synchronize()

        # Benchmark
        start_time = time.perf_counter()
        for _ in range(self.benchmark_runs):
            model(x)
            if self.device == "cuda":
                torch.cuda.synchronize()
        end_time = time.perf_counter()

        return (end_time - start_time) / self.benchmark_runs * 1000  # Convert to ms

    def measure_backward_time(self, model: nn.Module, x: torch.Tensor) -> float:
        """Measure backward pass time."""
        # Warmup
        for _ in range(self.warmup_runs):
            output = model(x)
            loss = output.sum()
            loss.backward()
            if self.device == "cuda":
                torch.cuda.synchronize()
            model.zero_grad()

        # Benchmark
        start_time = time.perf_counter()
        for _ in range(self.benchmark_runs):
            output = model(x)
            loss = output.sum()
            loss.backward()
            if self.device == "cuda":
                torch.cuda.synchronize()
            model.zero_grad()
        end_time = time.perf_counter()

        return (end_time - start_time) / self.benchmark_runs * 1000  # Convert to ms

    def measure_memory_usage(
        self, model: nn.Module, x: torch.Tensor
    ) -> dict[str, float]:
        """Measure memory usage during forward and backward pass."""
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            initial_memory = torch.cuda.memory_allocated()
            output = model(x)
            forward_memory = torch.cuda.memory_allocated()

            loss = output.sum()
            loss.backward()
            torch.cuda.synchronize()
            backward_memory = torch.cuda.max_memory_allocated()

            return {
                "forward_mb": (forward_memory - initial_memory) / 1024 / 1024,
                "backward_mb": (backward_memory - initial_memory) / 1024 / 1024,
            }
        else:
            # CPU memory tracking
            tracemalloc.start()

            output = model(x)
            _, forward_peak = tracemalloc.get_traced_memory()

            loss = output.sum()
            loss.backward()
            _, backward_peak = tracemalloc.get_traced_memory()

            tracemalloc.stop()

            return {
                "forward_mb": forward_peak / 1024 / 1024,
                "backward_mb": backward_peak / 1024 / 1024,
            }

    def benchmark_variant(
        self, variant_name: str, variant_class: type[nn.Module], config: dict[str, int]
    ) -> dict[str, float]:
        """Benchmark a single GLU variant."""
        model = variant_class(
            dim_in=config["dim_in"], dim_out=config["dim_out"], bias=True
        ).to(self.device)
        torch.compile(model)

        x = self.create_test_data(config["batch_size"], config["dim_in"])

        # Measure timing
        forward_time = self.measure_forward_time(model, x)
        backward_time = self.measure_backward_time(model, x)

        # Measure memory
        memory_stats = self.measure_memory_usage(model, x)

        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())

        return {
            "forward_time_ms": forward_time,
            "backward_time_ms": backward_time,
            "total_time_ms": forward_time + backward_time,
            "forward_memory_mb": memory_stats["forward_mb"],
            "backward_memory_mb": memory_stats["backward_mb"],
            "parameters": param_count,
        }

    def run_benchmarks(self) -> dict[str, dict[str, dict[str, float]]]:
        """Run benchmarks for all variants and configurations."""
        results = {}

        print(f"Running benchmarks on {self.device.upper()}")
        print("=" * 60)

        all_variants = {**self.glu_variants, **self.mglu_variants}

        for config in self.test_configs:
            config_name = (
                f"B{config['batch_size']}_I{config['dim_in']}_O{config['dim_out']}"
            )
            print(f"\nBenchmarking configuration: {config_name}")
            results[config_name] = {}

            for variant_name, variant_class in all_variants.items():
                print(f"  Testing {variant_name}...")
                try:
                    benchmark_result = self.benchmark_variant(
                        variant_name, variant_class, config
                    )
                    results[config_name][variant_name] = benchmark_result
                except Exception as e:
                    print(f"    Error benchmarking {variant_name}: {e}")
                    results[config_name][variant_name] = {"error": str(e)}

        return results

    def print_results_table(
        self, results: dict[str, dict[str, dict[str, float]]]
    ) -> None:
        """Print benchmark results in a formatted table."""
        print("\n" + "=" * 100)
        print("BENCHMARK RESULTS")
        print("=" * 100)

        for config_name, config_results in results.items():
            print(f"\nConfiguration: {config_name}")

            # Prepare table data
            headers = [
                "Variant",
                "Forward (ms)",
                "Backward (ms)",
                "Total (ms)",
                "Fwd Mem (MB)",
                "Bwd Mem (MB)",
                "Parameters",
            ]
            table_data = []

            # Sort variants for consistent output
            sorted_variants = []

            # Regular GLUs first
            for variant in ["GLU", "Bilinear", "ReGLU", "SwiGLU", "GeGLU"]:
                if variant in config_results:
                    sorted_variants.append(variant)

            # MGLUs second
            for variant in ["MGLU", "BilinearMGLU", "ReMGLU", "SwiMGLU", "GeMGLU"]:
                if variant in config_results:
                    sorted_variants.append(variant)

            for variant in sorted_variants:
                stats = config_results[variant]
                if "error" in stats:
                    table_data.append(
                        [variant, "ERROR", stats["error"], "", "", "", ""]
                    )
                else:
                    table_data.append(
                        [
                            variant,
                            f"{stats['forward_time_ms']:.2f}",
                            f"{stats['backward_time_ms']:.2f}",
                            f"{stats['total_time_ms']:.2f}",
                            f"{stats['forward_memory_mb']:.1f}",
                            f"{stats['backward_memory_mb']:.1f}",
                            f"{stats['parameters']:,}",
                        ]
                    )

            print(
                tabulate(
                    table_data,
                    headers=headers,
                    tablefmt="grid",
                    stralign="center",
                    numalign="center",
                )
            )

    def compare_glu_vs_mglu(
        self, results: dict[str, dict[str, dict[str, float]]]
    ) -> None:
        """Compare GLU vs MGLU variants."""
        print("\n" + "=" * 100)
        print("GLU vs MGLU COMPARISON")
        print("=" * 100)

        glu_mglu_pairs = [
            ("GLU", "MGLU"),
            ("Bilinear", "BilinearMGLU"),
            ("ReGLU", "ReMGLU"),
            ("SwiGLU", "SwiMGLU"),
            ("GeGLU", "GeMGLU"),
        ]

        for config_name, config_results in results.items():
            print(f"\nConfiguration: {config_name}")

            # Prepare comparison table data
            headers = [
                "Variant Pair",
                "Time Ratio (MGLU/GLU)",
                "Memory Ratio (MGLU/GLU)",
                "Param Ratio (MGLU/GLU)",
                "Performance",
            ]
            table_data = []

            for glu_name, mglu_name in glu_mglu_pairs:
                if glu_name in config_results and mglu_name in config_results:
                    glu_stats = config_results[glu_name]
                    mglu_stats = config_results[mglu_name]

                    if "error" not in glu_stats and "error" not in mglu_stats:
                        time_ratio = (
                            mglu_stats["total_time_ms"] / glu_stats["total_time_ms"]
                        )
                        memory_ratio = (
                            mglu_stats["backward_memory_mb"]
                            / glu_stats["backward_memory_mb"]
                        )
                        param_ratio = mglu_stats["parameters"] / glu_stats["parameters"]

                        # Determine performance indicator
                        if time_ratio < 1.0 and memory_ratio < 1.0:
                            performance = "✓ MGLU Better"
                        elif time_ratio > 1.0 and memory_ratio > 1.0:
                            performance = "✗ GLU Better"
                        else:
                            performance = "~ Mixed"

                        table_data.append(
                            [
                                f"{glu_name} / {mglu_name}",
                                f"{time_ratio:.3f}",
                                f"{memory_ratio:.3f}",
                                f"{param_ratio:.3f}",
                                performance,
                            ]
                        )

            if table_data:
                print(
                    tabulate(
                        table_data,
                        headers=headers,
                        tablefmt="grid",
                        stralign="center",
                        numalign="center",
                    )
                )


def main() -> None:
    """Main function to run the GLU benchmarks."""
    print("GLU Variants Profiling Script")
    print("=" * 40)

    # Check available devices
    # if torch.cuda.is_available():
    #     print(f"CUDA available: {torch.cuda.get_device_name()}")
    #     device = "cuda"
    # else:
    #     print("CUDA not available, using CPU")
    #     device = "cpu"
    device = "cpu"

    # Run benchmarks
    benchmark = GLUBenchmark(device=device)
    results = benchmark.run_benchmarks()

    # Print results
    benchmark.print_results_table(results)
    benchmark.compare_glu_vs_mglu(results)

    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("\nNotes:")
    print("- Time measurements are in milliseconds (ms)")
    print("- Memory measurements are in megabytes (MB)")
    print("- Parameters are formatted with thousands separators for readability")
    print(
        "- In comparison ratios: < 1.0 indicates MGLU is better, > 1.0 indicates GLU is better"
    )
    print(
        "- Performance indicators: ✓ (both time and memory better), ✗ (both worse), ~ (mixed results)"
    )


if __name__ == "__main__":
    main()
