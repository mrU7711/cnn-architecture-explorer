"""Performance benchmarking for CNN architectures."""

import time
import torch
import numpy as np
from src.architectures import count_parameters


def measure_inference_time(model, input_size=(1, 3, 224, 224), num_runs=100, device="cpu"):
    """Measure average inference time in milliseconds."""
    model = model.to(device)
    model.eval()
    dummy = torch.randn(*input_size).to(device)

    with torch.no_grad():
        for _ in range(10):
            model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(dummy)
            end = time.perf_counter()
            times.append((end - start) * 1000)

    return {"mean_ms": np.mean(times), "std_ms": np.std(times), "min_ms": np.min(times)}


def estimate_flops(model, input_size=(1, 3, 224, 224)):
    """Rough FLOPs estimation based on parameter count and layer types."""
    total_params = count_parameters(model, trainable_only=False)
    estimated_flops = total_params * 2
    return estimated_flops


def benchmark_model(model, name, device="cpu"):
    """Run all benchmarks on a model."""
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    timing = measure_inference_time(model, device=device)
    flops = estimate_flops(model)
    return {
        "name": name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "inference_ms": timing["mean_ms"],
        "estimated_flops": flops,
    }
