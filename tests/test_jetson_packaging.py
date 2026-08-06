"""Static validation for the JetPack 5 deployment path."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_jetson_installer_pins_xavier_runtime() -> None:
    requirements = (ROOT / "requirements.jetson-jp5.txt").read_text()
    installer = (ROOT / "install-jetson-jp5.sh").read_text()
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert "horde_safety~=0.3.0" in requirements
    assert "haidra-core==0.0.5" in requirements
    assert "mediapipe==0.10.18" in requirements
    assert "onnxruntime==1.17.3" in requirements
    assert "textual==8.1.1" in requirements
    assert "opentelemetry-instrumentation-httpx==0.48b0" in requirements
    assert "opentelemetry-instrumentation-system-metrics==0.48b0" in requirements
    assert "comfy-kitchen==" not in requirements
    requirement_names = {
        line.split("==", 1)[0].split("~=", 1)[0].split(">=", 1)[0].lower()
        for line in requirements.splitlines()
        if line and not line.startswith("#")
    }
    assert "torch" not in requirement_names
    assert "torchvision" not in requirement_names
    assert "torchaudio" not in requirement_names

    assert "${HOME}/.pyenv/versions/3.10.20/bin/python" in installer
    assert '"setuptools==80.9.0"' in installer
    assert "torch-2.1.0a0+git7bcf7da-cp310-cp310-linux_aarch64.whl" in installer
    assert "torchvision-0.16.0+fbb4cc5-cp310-cp310-linux_aarch64.whl" in installer
    assert "torchaudio-2.1.0+6ea1133-cp310-cp310-linux_aarch64.whl" in installer
    assert "triton-2.1.0+xavierjp5-cp310-cp310-linux_aarch64.whl" in installer
    assert "comfy_kitchen-0.2.26-py3-none-any.whl" in installer
    assert "flash_attn_legacy-0.5.0+xavierjp5-cp310-cp310-linux_aarch64.whl" in installer
    assert "xformers-0.0.23+e1b36f7.d*" in installer
    assert '"horde_engine>=3.0.0,<4.0.0"' in pyproject
    assert "https://github.com/Haidra-Org/hordelib.git" in installer
    assert "e4fbf6838e4d56b67f2173ab42a8eda368b601e7" in installer
    assert "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_HORDE_ENGINE=3.0.0" in installer
    assert "horde-engine-v3-e4fb-python310-jp5.patch" in installer
    assert "137213ad593b5ce970344f74db0b331c9031c966" in installer
    assert "horde-model-reference-v5.1.1-python310.patch" in installer
    assert "2fb40234105430449567265d3729887b86d21112" in installer
    assert "horde-sdk-v0.20.7-python310.patch" in installer
    assert 'backends["eager"]["available"]' in installer
    assert 'backends["triton"]["available"]' in installer
    assert 'for device in ("cpu", "cuda")' in installer
    assert 'git -C "$source" apply --ignore-space-change --whitespace=nowarn --check' in installer
    assert 'git -C "$source" ls-files --others --exclude-standard' in installer
    assert "CMAKE_BUILD_PARALLEL_LEVEL=1" in installer
    assert "MAKEFLAGS=-j1" in installer
    assert "MAX_JOBS=1" in installer
    assert '\n"$python" -m pip check\n' in installer
    assert 'metadata.version("horde-worker-regen") == "12.0.0"' in installer
    assert 'metadata.version("horde-sdk") == "0.20.7"' in installer
    assert 'metadata.version("horde-model-reference") == "5.1.1"' in installer
    assert 'metadata.version("triton") == "2.1.0+xavierjp5"' in installer
    assert 'metadata.version("comfy-kitchen") == "0.2.26"' in installer
    assert 'metadata.version("flash-attn-legacy") == "0.5.0+xavierjp5"' in installer


def test_jetson_installer_assets_are_bundled_with_patch_directory() -> None:
    manifest = (ROOT / "packaging" / "bundle-include.txt").read_text()
    release_workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()

    entries = {
        line.strip()
        for line in manifest.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert {
        "build-xformers-jetson-jp5.sh",
        "install-jetson-jp5.sh",
        "start-jetson-jp5.sh",
        "requirements.jetson-jp5.txt",
        "jetson-wheel-checksums.sha256",
        "jetson-patches",
    }.issubset(entries)
    assert 'cp -R "$f" "$stage/"' in release_workflow


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
    assert "triton-2.1.0+xavierjp5" in checksums
    assert "comfy_kitchen-0.2.26" in checksums
    assert "flash_attn_legacy-0.5.0+xavierjp5" in checksums
    assert "xformers-0.0.23+e1b36f7.d20260803" in checksums


def test_jetson_launcher_uses_bounded_cpu_threads() -> None:
    launcher = (ROOT / "start-jetson-jp5.sh").read_text()

    assert "getconf _NPROCESSORS_CONF" in launcher
    assert "compute_threads=$((configured_cpus / 2))" in launcher
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


def test_worker_avoids_python_311_strenum() -> None:
    """JetPack 5 runs Python 3.10, where stdlib enum.StrEnum is unavailable."""
    worker_sources = (ROOT / "horde_worker_regen").rglob("*.py")

    for source in worker_sources:
        assert "enum.StrEnum" not in source.read_text(), source
