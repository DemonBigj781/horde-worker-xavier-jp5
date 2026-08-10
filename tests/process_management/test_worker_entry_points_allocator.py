import os

import pytest

from horde_worker_regen.process_management import worker_entry_points


@pytest.fixture(autouse=True)
def clear_allocator_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep allocator assertions isolated from the developer's shell environment."""
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_HIP_ALLOC_CONF", raising=False)


def test_jetson_disables_expandable_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Jetson must avoid the PyTorch allocator path that requires desktop NVML."""
    monkeypatch.setattr(os.path, "exists", lambda path: path == "/etc/nv_tegra_release")

    worker_entry_points._enable_expandable_segments(amd_gpu=False, directml=None)

    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"


def test_jetson_preserves_operator_allocator_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit allocator policy remains authoritative on Jetson."""
    monkeypatch.setattr(os.path, "exists", lambda path: path == "/etc/nv_tegra_release")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    worker_entry_points._enable_expandable_segments(amd_gpu=False, directml=None)

    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"


def test_non_jetson_cuda_enables_expandable_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Desktop CUDA retains the upstream fragmentation guard."""
    monkeypatch.setattr(os.path, "exists", lambda _path: False)

    worker_entry_points._enable_expandable_segments(amd_gpu=False, directml=None)

    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_rocm_enables_expandable_segments_on_all_platforms(monkeypatch: pytest.MonkeyPatch) -> None:
    """ROCm does not inherit Jetson's NVIDIA-specific allocator restriction."""
    monkeypatch.setattr(os.path, "exists", lambda _path: True)

    worker_entry_points._enable_expandable_segments(amd_gpu=True, directml=None)

    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert os.environ["PYTORCH_HIP_ALLOC_CONF"] == "expandable_segments:True"
