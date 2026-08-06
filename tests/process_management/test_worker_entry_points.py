"""Tests for inference-process launch configuration."""

from horde_worker_regen.process_management.worker_entry_points import (
    _build_inference_comfyui_args,
    _build_models_not_to_force_load,
)


def test_shared_memory_reserve_enables_comfy_smart_memory() -> None:
    """A Xavier reserve requires ComfyUI's offload-aware smart-memory path."""
    args = _build_inference_comfyui_args(vram_reserve_gib=6)

    assert "--disable-smart-memory" not in args
    assert args[-2:] == ["--reserve-vram", "6"]


def test_shared_memory_reserve_allows_large_models_to_offload() -> None:
    """Guarded loading must not force-load SDXL or Cascade into unified memory."""
    models = _build_models_not_to_force_load(vram_reserve_gib=6)

    assert "stable_diffusion_xl" in models
    assert "stable_cascade" in models


def test_low_memory_mode_wins_over_shared_memory_reserve() -> None:
    """Explicit low-memory mode keeps its established ComfyUI behavior."""
    args = _build_inference_comfyui_args(low_memory_mode=True, vram_reserve_gib=6)

    assert "--novram" in args
    assert "--reserve-vram" not in args
