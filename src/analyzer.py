"""Architecture analysis and comparison tools."""

import pandas as pd
from src.architectures import load_architecture, count_parameters
from src.benchmark import benchmark_model


def compare_architectures(arch_configs, device="cpu"):
    """Compare multiple architectures and return results DataFrame."""
    results = []
    for config in arch_configs:
        name = config["name"]
        model = load_architecture(name, pretrained=config.get("pretrained", True),
                                   num_classes=config.get("num_classes", 10))
        benchmark = benchmark_model(model, name, device=device)
        results.append(benchmark)
    return pd.DataFrame(results)


def print_comparison(df):
    """Print formatted comparison table."""
    print("=" * 70)
    print(f"{'Model':<15} {'Params(M)':>10} {'Trainable(M)':>12} {'Infer(ms)':>10}")
    print("=" * 70)
    for _, row in df.iterrows():
        print(f"{row['name']:<15} {row['total_params']/1e6:>10.1f} "
              f"{row['trainable_params']/1e6:>12.1f} {row['inference_ms']:>10.1f}")
