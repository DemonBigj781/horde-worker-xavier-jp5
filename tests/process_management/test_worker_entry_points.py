"""Tests for inference-process launch configuration."""

from horde_worker_regen.process_management.worker_entry_points import (
    _build_inference_comfyui_args,
    _build_models_not_to_force_load,
    _is_jetson_runtime,
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


def test_flash_attention_can_be_selected_for_comfyui() -> None:
    args = _build_inference_comfyui_args(use_flash_attention=True)

    assert "--use-flash-attention" in args
    assert "--use-pytorch-cross-attention" not in args


def test_triton_backend_can_be_enabled_for_comfyui() -> None:
    args = _build_inference_comfyui_args(enable_triton_backend=True)

    assert "--enable-triton-backend" in args
    assert "--use-flash-attention" not in args


def test_amd_attention_wins_over_flash_attention() -> None:
    args = _build_inference_comfyui_args(
        amd_gpu=True,
        use_flash_attention=True,
        enable_triton_backend=True,
    )

    assert "--use-pytorch-cross-attention" in args
    assert "--use-flash-attention" not in args
    assert "--enable-triton-backend" in args


def test_jetson_runtime_detection(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    monkeypatch.setattr("os.path.isfile", lambda path: path == "/etc/nv_tegra_release")

    assert _is_jetson_runtime()
