#!/bin/sh
set -eu

script_dir=$(
	CDPATH=
	cd -P "$(dirname "$0")"
	pwd
)
wheel_dir=${JETSON_WHEEL_DIR:-${HOME}/jetson}
python_bin=${PYTHON_BIN:-${HOME}/.pyenv/versions/3.10.20/bin/python}
venv_dir=${JETSON_VENV_DIR:-$script_dir/.venv}
build_root=${JETSON_BUILD_ROOT:-/mnt/xavier-ssd/build}
engine_source=${JETSON_HORDE_ENGINE_SOURCE:-$build_root/tazlin-hordelib-a0555b4}
engine_commit=a0555b474696257a2374f4d1d4bc10b3d3fae5e3
checksum_file=$script_dir/jetson-wheel-checksums.sha256

torch_wheel=$wheel_dir/torch-2.1.0a0+git7bcf7da-cp310-cp310-linux_aarch64.whl
torchvision_wheel=$wheel_dir/torchvision-0.16.0+fbb4cc5-cp310-cp310-linux_aarch64.whl
torchaudio_wheel=$wheel_dir/torchaudio-2.1.0+6ea1133-cp310-cp310-linux_aarch64.whl

test "$(uname -m)" = aarch64 || {
	printf '%s\n' 'JetPack 5 support requires an aarch64 host.' >&2
	exit 1
}
if ! test -r /etc/nv_tegra_release || ! grep -q '^# R35 ' /etc/nv_tegra_release; then
	printf '%s\n' 'JetPack 5 support requires L4T R35.' >&2
	exit 1
fi
test -x "$python_bin" || {
	printf 'Python 3.10.20 was not found at %s\n' "$python_bin" >&2
	exit 1
}
python_version=$("$python_bin" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
test "$python_version" = 3.10.20 || {
	printf 'Expected Python 3.10.20, found %s\n' "$python_version" >&2
	exit 1
}
test -f "$checksum_file" || {
	printf 'Wheel checksum manifest not found: %s\n' "$checksum_file" >&2
	exit 1
}

set -- "$wheel_dir"/xformers-0.0.23+e1b36f7.d*-cp310-cp310-linux_aarch64.whl
test "$#" -eq 1 && test -f "$1" || {
	printf '%s\n' 'Expected exactly one tested Xavier xFormers wheel.' >&2
	exit 1
}
xformers_wheel=$1
xformers_sidecar=$xformers_wheel.sha256
test -f "$xformers_sidecar" || {
	printf 'xFormers .sha256 sidecar not found: %s\n' "$xformers_sidecar" >&2
	exit 1
}

for wheel in "$torch_wheel" "$torchvision_wheel" "$torchaudio_wheel" "$xformers_wheel"; do
	test -f "$wheel" || {
		printf 'Required Jetson wheel not found: %s\n' "$wheel" >&2
		exit 1
	}
done

(
	cd "$wheel_dir"
	sha256sum -c "$checksum_file"
	sha256sum -c "$(basename "$xformers_sidecar")"
)

test ! -e "$venv_dir" || {
	printf 'Refusing to overwrite existing virtual environment: %s\n' "$venv_dir" >&2
	exit 1
}

export CMAKE_BUILD_PARALLEL_LEVEL=1
export CARGO_BUILD_JOBS=1
export MAX_JOBS=1
export NINJAFLAGS=-j1
export NVCC_THREADS=1

"$python_bin" -m venv "$venv_dir"
python=$venv_dir/bin/python

"$python" -m pip install --upgrade pip "setuptools==80.9.0" wheel
"$python" -m pip install --no-deps "$torch_wheel" "$torchvision_wheel" "$torchaudio_wheel"
"$python" -m pip install -r "$script_dir/requirements.jetson-jp5.txt"

if test -e "$engine_source"; then
	test -d "$engine_source/.git" || {
		printf 'Existing horde-engine source is not a Git checkout: %s\n' "$engine_source" >&2
		exit 1
	}
	test "$(git -C "$engine_source" rev-parse HEAD)" = "$engine_commit" || {
		printf 'Existing horde-engine source is not pinned to %s\n' "$engine_commit" >&2
		exit 1
	}
else
	git clone --recursive https://github.com/tazlin/hordelib.git "$engine_source"
	git -C "$engine_source" switch --detach "$engine_commit"
	git -C "$engine_source" submodule update --init --recursive
fi

SETUPTOOLS_SCM_PRETEND_VERSION_FOR_HORDE_ENGINE=2.20.12 \
	"$python" -m pip install --no-deps "$engine_source"
"$python" -m pip install --no-deps "$xformers_wheel"
"$python" -m pip install --no-deps -e "$script_dir"

# The worker metadata follows modern Torch and MediaPipe pins. JetPack 5 must
# retain NVIDIA's CUDA 11.4 build and the newest Python 3.10 aarch64 MediaPipe
# wheel, but no other dependency mismatch is allowed.
pip_check_output=$("$python" -m pip check 2>&1 || true)
unexpected_pip_check_output=$(
	printf '%s\n' "$pip_check_output" | grep -Fv \
		-e "No broken requirements found." \
		-e "horde-worker-regen 10.1.2 has requirement mediapipe==0.10.21, but you have mediapipe 0.10.18." \
		-e "horde-worker-regen 10.1.2 has requirement torch==2.9.1, but you have torch 2.1.0a0+git7bcf7da." || true
)
if test -n "$unexpected_pip_check_output"; then
	printf '%s\n' "$pip_check_output" >&2
	exit 1
fi

"$python" -s - <<'PY'
import importlib.metadata as metadata

import horde_model_reference
import horde_safety
import horde_sdk
import hordelib
import torch
import xformers

assert torch.__version__ == "2.1.0a0+git7bcf7da", torch.__version__
assert torch.version.cuda == "11.4", torch.version.cuda
assert torch.cuda.is_available(), "PyTorch cannot access Xavier's CUDA device"
assert torch.cuda.get_device_capability(0) == (7, 2)
assert xformers._has_cpp_library, "xFormers CUDA extension is unavailable"
assert metadata.version("horde-worker-regen") == "10.1.2"
assert metadata.version("horde-engine") == "2.20.12"
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
print("xformers", metadata.version("xformers"))
print("worker", metadata.version("horde-worker-regen"))
print("engine", metadata.version("horde-engine"))
print("horde imports", hordelib.__name__, horde_sdk.__name__, horde_safety.__name__, horde_model_reference.__name__)
PY
