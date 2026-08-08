#!/usr/bin/env python3
"""Run offline xFormers and legacy FlashAttention checks on Xavier SM72."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Callable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Xavier's xFormers and legacy FlashAttention paths without starting a worker.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260808)
    return parser


def _read_kib(path: Path, key: str) -> int:
    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition(":")
        if separator and name == key:
            return int(value.split()[0])
    raise RuntimeError(f"{key} was not found in {path}")


def _host_memory_snapshot() -> dict[str, int]:
    return {
        "available_ram_kib": _read_kib(Path("/proc/meminfo"), "MemAvailable"),
        "process_rss_kib": _read_kib(Path("/proc/self/status"), "VmRSS"),
    }


def _measure_backend(
    torch: Any,
    operation: Callable[[], Any],
    reference: Any,
    iterations: int,
) -> dict[str, float | int]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    operation()
    torch.cuda.synchronize()

    started = time.perf_counter()
    output = None
    for _ in range(iterations):
        output = operation()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert output is not None
    difference = (output.float() - reference.float()).abs()
    return {
        "iterations": iterations,
        "mean_ms": elapsed_ms / iterations,
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def run(args: argparse.Namespace) -> int:
    if min(args.batch_size, args.sequence_length, args.heads, args.head_dim, args.iterations) <= 0:
        raise ValueError("shape dimensions and iterations must be positive")

    import torch
    import torch.nn.functional as functional
    from flash_attn import flash_attn_func
    from xformers.ops import memory_efficient_attention

    if not torch.cuda.is_available():
        raise RuntimeError("Xavier attention validation requires CUDA")
    capability = torch.cuda.get_device_capability(0)
    if capability != (7, 2):
        raise RuntimeError(f"expected Xavier SM72, found compute capability {capability}")

    flash_providers = metadata.packages_distributions().get("flash_attn")
    if flash_providers != ["flash-attn-legacy"]:
        raise RuntimeError(
            "flash_attn must be provided only by flash-attn-legacy; "
            f"found {flash_providers!r}",
        )

    torch.manual_seed(args.seed)
    shape = (args.batch_size, args.sequence_length, args.heads, args.head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.1
    k = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.1
    v = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.1
    scale = 1.0 / math.sqrt(args.head_dim)
    reference = functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=scale,
    ).transpose(1, 2).contiguous()
    torch.cuda.synchronize()

    before = _host_memory_snapshot()
    xformers_metrics = _measure_backend(
        torch,
        lambda: memory_efficient_attention(q, k, v, scale=scale),
        reference,
        args.iterations,
    )
    flash_metrics = _measure_backend(
        torch,
        lambda: flash_attn_func(q, k, v, softmax_scale=scale),
        reference,
        args.iterations,
    )
    after = _host_memory_snapshot()

    max_allowed_error = 0.004
    for backend, metrics in (("xformers", xformers_metrics), ("flash-attn-legacy", flash_metrics)):
        if metrics["max_abs_error"] > max_allowed_error:
            raise RuntimeError(
                f"{backend} exceeded the error limit: "
                f"{metrics['max_abs_error']:.8f} > {max_allowed_error:.8f}",
            )

    report = {
        "status": "passed",
        "runtime": {
            "python": os.sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(capability),
            "xformers": metadata.version("xformers"),
            "flash_attn_legacy": metadata.version("flash-attn-legacy"),
            "flash_attn_providers": flash_providers,
        },
        "shape": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "dtype": "float16",
            "seed": args.seed,
        },
        "host_memory_before": before,
        "host_memory_after": after,
        "backends": {
            "xformers": xformers_metrics,
            "flash-attn-legacy": flash_metrics,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("XAVIER_ATTENTION_COMPAT_SUCCESS " + json.dumps(report, sort_keys=True), flush=True)
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
