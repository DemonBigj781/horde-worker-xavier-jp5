#!/usr/bin/env python3
"""Run FLUX with explicit conditioning, denoising, and tiled-decode lifetimes."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT_NAME = "flux1CompactCLIPAnd_Flux1SchnellFp8.safetensors"
DEFAULT_PROMPT = "A compact futuristic research rover on a red desert plain, technical photography"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    cache_home = Path(os.environ.get("AIWORKER_CACHE_HOME", "models"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=cache_home / "compvis" / DEFAULT_CHECKPOINT_NAME,
        help="FLUX checkpoint path",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser


def read_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in path.read_text().splitlines():
            key, separator, remainder = line.partition(":")
            if not separator:
                continue
            fields = remainder.split()
            if fields and fields[0].isdigit():
                values[key] = int(fields[0])
    except OSError:
        pass
    return values


def memory_snapshot(torch: Any, label: str) -> dict[str, Any]:
    meminfo = read_key_values(Path("/proc/meminfo"))
    status = read_key_values(Path("/proc/self/status"))
    snapshot = {
        "label": label,
        "monotonic_seconds": time.monotonic(),
        "mem_available_kib": meminfo.get("MemAvailable"),
        "mem_free_kib": meminfo.get("MemFree"),
        "cached_kib": meminfo.get("Cached"),
        "swap_free_kib": meminfo.get("SwapFree"),
        "process_rss_kib": status.get("VmRSS"),
        "process_hwm_kib": status.get("VmHWM"),
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    print("MEMORY_SNAPSHOT " + json.dumps(snapshot, sort_keys=True), flush=True)
    return snapshot


def release_component_memory(torch: Any, model_management: Any, label: str) -> None:
    try:
        torch.cuda.synchronize()
    except RuntimeError as error:
        print(f"CUDA_SYNC_WARNING label={label} error={error}", flush=True)
    model_management.unload_all_models()
    model_management.soft_empty_cache(True)
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError as error:
        print(f"CUDA_IPC_WARNING label={label} error={error}", flush=True)
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError) as error:
        print(f"MALLOC_TRIM_WARNING label={label} error={error}", flush=True)
    time.sleep(2)


def save_image(images: Any, output_path: Path) -> None:
    import numpy as np
    from PIL import Image

    pixels = images[0].detach().to("cpu").clamp(0.0, 1.0).mul(255).byte().numpy()
    Image.fromarray(np.asarray(pixels)).save(output_path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    import torch
    from hordelib.api import initialise

    initialise(
        setup_logging=True,
        logging_verbosity=4,
        force_normal_vram_mode=True,
        disable_smart_memory=False,
        do_not_load_model_mangers=True,
    )

    import comfy.model_management as model_management
    import comfy.sd
    import folder_paths
    from comfy_extras.nodes_custom_sampler import (
        BasicScheduler,
        CFGGuider,
        KSamplerSelect,
        RandomNoise,
        SamplerCustomAdvanced,
    )
    from nodes import CLIPTextEncode, EmptyLatentImage, VAEDecodeTiled

    metrics: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "snapshots": [],
        "stages": {},
        "success": False,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "compute_capability": torch.cuda.get_device_capability(),
        },
    }
    output_path = args.output_dir / f"flux-staged-{args.width}x{args.height}-tile{args.tile_size}.png"
    torch.cuda.reset_peak_memory_stats()
    metrics["snapshots"].append(memory_snapshot(torch, "initialized"))

    stage_started = time.monotonic()
    loaded_checkpoint = comfy.sd.load_checkpoint_guess_config(
        str(args.checkpoint),
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    model, clip, vae = loaded_checkpoint[:3]
    del loaded_checkpoint
    metrics["snapshots"].append(memory_snapshot(torch, "checkpoint_loaded"))
    encoder = CLIPTextEncode()
    positive = encoder.encode(clip, args.prompt)[0]
    negative = encoder.encode(clip, "")[0]
    metrics["snapshots"].append(memory_snapshot(torch, "conditioning_complete"))
    del encoder, clip
    release_component_memory(torch, model_management, "after_conditioning")
    metrics["snapshots"].append(memory_snapshot(torch, "conditioning_released"))
    metrics["stages"]["conditioning_seconds"] = time.monotonic() - stage_started

    stage_started = time.monotonic()
    latent_image = EmptyLatentImage().generate(args.width, args.height, 1)[0]
    sigmas = BasicScheduler.get_sigmas(model, "normal", args.steps, 1.0)[0]
    sampler = KSamplerSelect.get_sampler("euler")[0]
    noise = RandomNoise.get_noise(args.seed)[0]
    guider = CFGGuider.get_guider(model, positive, negative, 1.0)[0]
    latent = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)[0]
    torch.cuda.synchronize()
    metrics["snapshots"].append(memory_snapshot(torch, "denoising_complete"))
    del guider, noise, sampler, sigmas, latent_image, positive, negative, model
    release_component_memory(torch, model_management, "after_denoising")
    metrics["snapshots"].append(memory_snapshot(torch, "transformer_released"))
    metrics["stages"]["denoising_seconds"] = time.monotonic() - stage_started

    stage_started = time.monotonic()
    images = VAEDecodeTiled().decode(
        vae,
        latent,
        tile_size=args.tile_size,
        overlap=args.tile_overlap,
    )[0]
    torch.cuda.synchronize()
    metrics["snapshots"].append(memory_snapshot(torch, "decode_complete"))
    save_image(images, output_path)
    metrics["stages"]["decode_seconds"] = time.monotonic() - stage_started
    del images, latent, vae
    release_component_memory(torch, model_management, "after_decode")
    metrics["snapshots"].append(memory_snapshot(torch, "all_components_released"))

    metrics["success"] = True
    metrics["output_path"] = str(output_path)
    metrics["output_bytes"] = output_path.stat().st_size
    metrics["output_sha256"] = sha256_file(output_path)
    metrics["total_seconds"] = sum(metrics["stages"].values())
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("FLUX_STAGED_TEST_SUCCESS " + json.dumps(metrics, sort_keys=True), flush=True)
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as error:
        print(f"FLUX_STAGED_TEST_FAILURE {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
    raise SystemExit(exit_code)
