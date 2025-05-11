"""Unit tests for benchmarking."""

import pytest
import torch
from src.architectures import load_architecture
from src.benchmark import measure_inference_time, benchmark_model


class TestBenchmark:
    def test_inference_time(self):
        model = load_architecture("resnet18", pretrained=False)
        timing = measure_inference_time(model, num_runs=5)
        assert timing["mean_ms"] > 0

    def test_benchmark_model(self):
        model = load_architecture("resnet18", pretrained=False)
        result = benchmark_model(model, "resnet18")
        assert "total_params" in result
        assert "inference_ms" in result
        assert result["total_params"] > 0
