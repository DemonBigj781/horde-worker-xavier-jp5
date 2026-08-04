from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_jetson_installer_pins_xavier_runtime() -> None:
    requirements = (ROOT / "requirements.jetson-jp5.txt").read_text()
    installer = (ROOT / "install-jetson-jp5.sh").read_text()

    assert "horde_sdk~=0.17.1" in requirements
    assert "horde_safety~=0.3.0" in requirements
    assert "horde_model_reference~=0.9.2" in requirements
    assert "onnxruntime==1.17.3" in requirements
    requirement_names = {
        line.split("==", 1)[0].split("~=", 1)[0].split(">=", 1)[0].lower()
        for line in requirements.splitlines()
        if line and not line.startswith("#")
    }
    assert "torch" not in requirement_names
    assert "torchvision" not in requirement_names
    assert "torchaudio" not in requirement_names

    assert "${HOME}/.pyenv/versions/3.10.20/bin/python" in installer
    assert "torch-2.1.0a0+git7bcf7da-cp310-cp310-linux_aarch64.whl" in installer
    assert "torchvision-0.16.0+fbb4cc5-cp310-cp310-linux_aarch64.whl" in installer
    assert "torchaudio-2.1.0+6ea1133-cp310-cp310-linux_aarch64.whl" in installer
    assert "xformers-0.0.23+e1b36f7.d*" in installer
    assert "a0555b474696257a2374f4d1d4bc10b3d3fae5e3" in installer
    assert "CMAKE_BUILD_PARALLEL_LEVEL=1" in installer
    assert "MAX_JOBS=1" in installer
    assert 'metadata.version("horde-worker-regen") == "10.1.2"' in installer


def test_jetson_installer_rejects_wrong_platform_and_unverified_wheels() -> None:
    installer = (ROOT / "install-jetson-jp5.sh").read_text()
    checksums = (ROOT / "jetson-wheel-checksums.sha256").read_text()

    assert 'test "$(uname -m)" = aarch64' in installer
    assert "^# R35 " in installer
    assert "3.10.20" in installer
    assert ".sha256" in installer
    assert "sha256sum -c" in installer
    assert '"$python" -m pip install --no-deps' in installer
    assert "torch-2.1.0a0+git7bcf7da" in checksums
    assert "torchvision-0.16.0+fbb4cc5" in checksums
    assert "torchaudio-2.1.0+6ea1133" in checksums
    assert "xformers-0.0.23+e1b36f7.d20260803" in checksums


def test_jetson_launcher_uses_bounded_cpu_threads() -> None:
    launcher = (ROOT / "start-jetson-jp5.sh").read_text()

    assert "getconf _NPROCESSORS_CONF" in launcher
    assert "compute_threads=$((configured_cpus / 2))" in launcher
    assert "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128" in launcher
    assert "PYTORCH_CHANNELS_LAST=1" in launcher
    assert "PYTORCH_JIT=0" in launcher
    assert "LD_PRELOAD" in launcher
    assert 'exec "$script_dir/.venv/bin/python" -s' in launcher


def test_xformers_builder_is_xavier_specific_and_single_threaded() -> None:
    builder = (ROOT / "build-xformers-jetson-jp5.sh").read_text()

    assert "e1b36f781ba1c9d10f36fc0ec87170e0b381fdad" in builder
    assert "TORCH_CUDA_ARCH_LIST=7.2" in builder
    assert "CUDA_HOME=/usr/local/cuda-11.4" in builder
    assert "XFORMERS_DISABLE_FLASH_ATTN=1" in builder
    assert "MAX_JOBS=1" in builder
    assert "CMAKE_BUILD_PARALLEL_LEVEL=1" in builder
    assert "NVCC_THREADS=1" in builder
    assert "NINJAFLAGS=-j1" in builder
