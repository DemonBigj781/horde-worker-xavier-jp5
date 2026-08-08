"""Static validation for the JetPack 5 dependency overlay."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _requirement_names(requirements: str) -> set[str]:
    return {
        line.split("==", 1)[0]
        .split("~=", 1)[0]
        .split(">=", 1)[0]
        .split("<", 1)[0]
        .strip()
        .lower()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_jetson_installer_pins_v13_dependency_contract() -> None:
    requirements = (ROOT / "requirements.jetson-jp5.txt").read_text(encoding="utf-8")
    constraints = (ROOT / "constraints.jetson-jp5.txt").read_text(encoding="utf-8")
    installer = (ROOT / "install-jetson-jp5.sh").read_text(encoding="utf-8")

    assert "-c constraints.jetson-jp5.txt" in requirements
    assert "horde_safety~=0.3.0" in requirements
    assert "mediapipe==0.10.18" in requirements
    assert "onnxruntime==1.23.2" in requirements
    assert "numpy==1.26.4" in requirements
    assert "protobuf<5" in requirements
    assert "comfy-kitchen==" not in requirements
    assert "matplotlib==3.10.9" in requirements
    assert "opencv-python==4.11.0.86" in requirements
    assert "opencv-contrib-python==4.11.0.86" in requirements
    assert "scikit-image==0.25.2" in requirements
    assert "scipy==1.15.3" in requirements
    assert "scikit-learn==1.7.2" in requirements
    assert "logfire==4.6.0" in requirements
    assert "opentelemetry-instrumentation-aiohttp-client==0.48b0" in requirements
    assert "opentelemetry-instrumentation-httpx==0.48b0" in requirements
    assert "opentelemetry-instrumentation-requests==0.48b0" in requirements
    assert "opentelemetry-instrumentation-system-metrics==0.48b0" in requirements

    assert "torch==2.1.0a0+git7bcf7da" in constraints
    assert "torchvision==0.16.0+fbb4cc5" in constraints
    assert "torchaudio==2.1.0+6ea1133" in constraints
    assert "opencv-python-headless==4.11.0.86" in constraints
    assert "opentelemetry-api==1.27.0" in constraints
    assert "opentelemetry-sdk==1.27.0" in constraints
    assert "opentelemetry-proto==1.27.0" in constraints
    assert "opentelemetry-semantic-conventions==0.48b0" in constraints
    assert "huggingface-hub==0.36.2" in constraints
    assert "importlib-metadata==8.4.0" in constraints

    requirement_names = _requirement_names(requirements)
    assert "torch" not in requirement_names
    assert "torchvision" not in requirement_names
    assert "torchaudio" not in requirement_names
    assert "horde-engine" not in requirement_names
    assert "horde-sdk" not in requirement_names
    assert "horde-model-reference" not in requirement_names

    assert "55cbba57dcc4ad06a379c2d07b9b2ff94b8cc775" in installer
    assert "horde-engine-v3.11.0-python310-jp5.patch" in installer
    assert "7c318a0a48534602b1825ed4c73e977c7f10e581" in installer
    assert "horde-sdk-v0.22.4-python310.patch" in installer
    assert "89c3a020d0818a9275f82f4b0b75d60cdd04a53f" in installer
    assert "horde-model-reference-v7.0.2-python310.patch" in installer

    assert 'metadata.version("horde-worker-regen") == "13.16.7"' in installer
    assert 'metadata.version("horde-engine") == "3.11.0"' in installer
    assert 'metadata.version("horde-sdk") == "0.22.4"' in installer
    assert 'metadata.version("horde-model-reference") == "7.0.2"' in installer
    assert 'metadata.version("vtracer") == "0.6.15"' in installer
    assert 'metadata.version("logfire") == "4.6.0"' in installer
    assert 'metadata.version("opentelemetry-api") == "1.27.0"' in installer
    assert 'metadata.version("opentelemetry-proto") == "1.27.0"' in installer
    assert 'metadata.version("protobuf") == "4.25.9"' in installer
    assert 'metadata.version("importlib-metadata") == "8.4.0"' in installer
    assert "Rust and Cargo are recommended" in installer
    assert "Rust cargo is required" not in installer


def test_horde_engine_patch_preserves_protobuf4_telemetry_stack() -> None:
    patch = (
        ROOT / "jetson-patches" / "horde-engine-v3.11.0-python310-jp5.patch"
    ).read_text(encoding="utf-8")

    assert '+    "opentelemetry-instrumentation-aiohttp-client==0.48b0",' in patch
    assert '+    "opentelemetry-instrumentation-requests==0.48b0",' in patch
    assert '+    "opentelemetry-instrumentation-aiohttp-client>=0.59b0",' not in patch
    assert '+    "opentelemetry-instrumentation-requests>=0.59b0",' not in patch


def test_scikit_learn_wheel_uses_xavier_system_openmp() -> None:
    installer = (ROOT / "install-jetson-jp5.sh").read_text(encoding="utf-8")
    repairer = (
        ROOT / "packaging" / "jetson" / "repair_scikit_learn_wheel.py"
    ).read_text(encoding="utf-8")

    wheel = (
        "scikit_learn-1.7.2-1xavierjp5-cp310-cp310-"
        "manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
    )
    assert wheel in installer
    assert "libgomp-947d5fa1.so.1.0.0" in repairer
    assert '"--replace-needed", private_libgomp, "libgomp.so.1"' in repairer
    assert "EXPECTED_PATCHED_EXTENSIONS = 18" in repairer
    assert 'metadata.version("scikit-learn") == "1.7.2"' in installer
    assert 'root.glob("scikit_learn.libs/libgomp*")' in installer


def test_jetson_installer_preserves_xavier_native_wheels() -> None:
    installer = (ROOT / "install-jetson-jp5.sh").read_text(encoding="utf-8")
    checksums = (ROOT / "jetson-wheel-checksums.sha256").read_text(encoding="utf-8")

    expected_wheels = (
        "torch-2.1.0a0+git7bcf7da-cp310-cp310-linux_aarch64.whl",
        "torchvision-0.16.0+fbb4cc5-cp310-cp310-linux_aarch64.whl",
        "torchaudio-2.1.0+6ea1133-cp310-cp310-linux_aarch64.whl",
        "triton-2.1.0+xavierjp5-cp310-cp310-linux_aarch64.whl",
        "comfy_kitchen-0.2.26-py3-none-any.whl",
        "flash_attn_legacy-0.5.0+xavierjp5-cp310-cp310-linux_aarch64.whl",
        "xformers-0.0.23+e1b36f7.d20260803-cp310-cp310-linux_aarch64.whl",
        (
            "scikit_learn-1.7.2-1xavierjp5-cp310-cp310-"
            "manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
        ),
    )
    for wheel in expected_wheels:
        assert wheel in installer or wheel in checksums

    assert 'test "$(uname -m)" = aarch64' in installer
    assert "^# R35 " in installer
    assert "3.10.20" in installer
    assert "sha256sum -c" in installer
    assert '"$python" -m pip install --no-deps' in installer
    assert "cu126" not in installer
    assert "cu130" not in installer
    assert "cu132" not in installer


def test_jetson_build_and_install_are_single_threaded() -> None:
    installer = (ROOT / "install-jetson-jp5.sh").read_text(encoding="utf-8")
    builder = (ROOT / "build-xformers-jetson-jp5.sh").read_text(encoding="utf-8")

    for script in (installer, builder):
        assert "CMAKE_BUILD_PARALLEL_LEVEL=1" in script
        assert "CARGO_BUILD_JOBS=1" in script
        assert "MAKEFLAGS=-j1" in script
        assert "MAX_JOBS=1" in script
        assert "NINJAFLAGS=-j1" in script
        assert "NVCC_THREADS=1" in script

    assert "TORCH_CUDA_ARCH_LIST=7.2" in builder
    assert "CUDA_HOME=/usr/local/cuda-11.4" in builder
    assert "XFORMERS_DISABLE_FLASH_ATTN=1" in builder


def test_jetson_assets_are_in_release_bundle() -> None:
    manifest = (ROOT / "packaging" / "bundle-include.txt").read_text(encoding="utf-8")
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
        "constraints.jetson-jp5.txt",
        "jetson-wheel-checksums.sha256",
        "jetson-patches",
        "packaging/jetson/repair_scikit_learn_wheel.py",
    }.issubset(entries)


def test_worker_and_patched_dependencies_parse_as_python_310() -> None:
    source_roots = (ROOT / "horde_worker_regen", ROOT / "worker_bootstrap")

    for source_root in source_roots:
        for source in source_root.rglob("*.py"):
            ast.parse(
                source.read_text(encoding="utf-8"),
                filename=str(source),
                feature_version=(3, 10),
            )
