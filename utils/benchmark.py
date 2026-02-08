"""Benchmarking utilities for model efficiency evaluation.

Measures parameter count, FLOPs, inference latency, and throughput.
"""

import time

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters.

    Args:
        model: PyTorch model.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Estimate model size in megabytes.

    Args:
        model: PyTorch model.

    Returns:
        Model size in MB.
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def measure_latency(
    model: nn.Module,
    input_size: tuple = (1, 3, 224, 224),
    device: str = "cuda",
    warmup_runs: int = 10,
    num_runs: int = 100,
) -> float:
    """Measure average inference latency in milliseconds.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape (batch=1 for latency).
        device: Device to run on ('cuda' or 'cpu').
        warmup_runs: Number of warmup forward passes.
        num_runs: Number of timed forward passes.

    Returns:
        Average latency in milliseconds.
    """
    model = model.to(device).eval()
    x = torch.randn(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

    return sum(times) / len(times)


def measure_throughput(
    model: nn.Module,
    input_size: tuple = (32, 3, 224, 224),
    device: str = "cuda",
    num_runs: int = 50,
) -> float:
    """Measure inference throughput in images/second.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape (use typical batch size).
        device: Device to run on.
        num_runs: Number of timed forward passes.

    Returns:
        Throughput in images per second.
    """
    model = model.to(device).eval()
    batch_size = input_size[0]
    x = torch.randn(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (batch_size * num_runs) / elapsed


def compute_flops(model: nn.Module, input_size: tuple = (1, 3, 224, 224)):
    """Compute FLOPs using fvcore or ptflops.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape.

    Returns:
        FLOPs count (float) or None if libraries are not available.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        x = torch.randn(*input_size)
        flops = FlopCountAnalysis(model.cpu(), x)
        return flops.total()
    except ImportError:
        pass

    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model.cpu(), input_size[1:], as_strings=False,
            print_per_layer_stat=False,
        )
        return macs * 2  # MACs to FLOPs
    except ImportError:
        pass

    return None
