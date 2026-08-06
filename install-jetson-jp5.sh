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
engine_source=${JETSON_HORDE_ENGINE_SOURCE:-$build_root/horde-engine-v3-e4fb-jp5}
engine_repository=https://github.com/Haidra-Org/hordelib.git
engine_commit=e4fbf6838e4d56b67f2173ab42a8eda368b601e7
engine_patch=$script_dir/jetson-patches/horde-engine-v3-e4fb-python310-jp5.patch
engine_manifest=$script_dir/jetson-patches/horde-engine-v3-e4fb-python310-jp5.sha256
engine_base_replacement=$script_dir/jetson-patches/horde-engine-base.py
model_reference_source=${JETSON_HORDE_MODEL_REFERENCE_SOURCE:-$build_root/horde-model-reference-jetson-py310-v5.1.1}
model_reference_repository=https://github.com/Haidra-Org/horde-model-reference.git
model_reference_commit=137213ad593b5ce970344f74db0b331c9031c966
model_reference_patch=$script_dir/jetson-patches/horde-model-reference-v5.1.1-python310.patch
model_reference_manifest=$script_dir/jetson-patches/horde-model-reference-v5.1.1-python310.sha256
sdk_source=${JETSON_HORDE_SDK_SOURCE:-$build_root/horde-sdk-jetson-py310-v0.20.7}
sdk_repository=https://github.com/Haidra-Org/horde-sdk.git
sdk_commit=2fb40234105430449567265d3729887b86d21112
sdk_patch=$script_dir/jetson-patches/horde-sdk-v0.20.7-python310.patch
sdk_manifest=$script_dir/jetson-patches/horde-sdk-v0.20.7-python310.sha256
checksum_file=$script_dir/jetson-wheel-checksums.sha256

torch_wheel=$wheel_dir/torch-2.1.0a0+git7bcf7da-cp310-cp310-linux_aarch64.whl
torchvision_wheel=$wheel_dir/torchvision-0.16.0+fbb4cc5-cp310-cp310-linux_aarch64.whl
torchaudio_wheel=$wheel_dir/torchaudio-2.1.0+6ea1133-cp310-cp310-linux_aarch64.whl
triton_wheel=$wheel_dir/triton-2.1.0+xavierjp5-cp310-cp310-linux_aarch64.whl
comfy_kitchen_wheel=$wheel_dir/comfy_kitchen-0.2.26-py3-none-any.whl
flash_attention_wheel=$wheel_dir/flash_attn_legacy-0.5.0+xavierjp5-cp310-cp310-linux_aarch64.whl

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
for backport_file in \
	"$engine_patch" "$engine_manifest" \
	"$engine_base_replacement" \
	"$model_reference_patch" "$model_reference_manifest" \
	"$sdk_patch" "$sdk_manifest"; do
	test -f "$backport_file" || {
		printf 'Backport file not found: %s\n' "$backport_file" >&2
		exit 1
	}
done

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

for wheel in \
	"$torch_wheel" "$torchvision_wheel" "$torchaudio_wheel" \
	"$triton_wheel" "$comfy_kitchen_wheel" "$flash_attention_wheel" \
	"$xformers_wheel"; do
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
export MAKEFLAGS=-j1
export MAX_JOBS=1
export NINJAFLAGS=-j1
export NVCC_THREADS=1

"$python_bin" -m venv "$venv_dir"
python=$venv_dir/bin/python

"$python" -m pip install --upgrade pip "setuptools==80.9.0" wheel
"$python" -m pip install --no-deps "$torch_wheel" "$torchvision_wheel" "$torchaudio_wheel"
"$python" -m pip install -r "$script_dir/requirements.jetson-jp5.txt"

write_source_file_list() {
	source=$1
	output=$2
	{
		git -C "$source" diff --name-only -- .
		git -C "$source" ls-files --others --exclude-standard
	} | LC_ALL=C sort -u >"$output"
}

ensure_patched_source() {
	name=$1
	repository=$2
	source=$3
	commit=$4
	patch_file=$5
	manifest_file=$6
	replacement_file=${7-}
	replacement_path=${8-}
	current_files=$(mktemp)
	expected_files=$(mktemp)

	if test -e "$source"; then
		test -d "$source/.git" || {
			printf 'Existing %s source is not a Git checkout: %s\n' "$name" "$source" >&2
			exit 1
		}
	else
		git clone --filter=blob:none "$repository" "$source"
	fi

	git -C "$source" switch --detach "$commit"
	test "$(git -C "$source" rev-parse HEAD)" = "$commit" || {
		printf '%s source is not pinned to %s\n' "$name" "$commit" >&2
		exit 1
	}

	write_source_file_list "$source" "$current_files"
	if test ! -s "$current_files"; then
		git -C "$source" apply --ignore-space-change --whitespace=nowarn --check "$patch_file"
		git -C "$source" apply --ignore-space-change --whitespace=nowarn "$patch_file"
	fi
	if test -n "$replacement_file"; then
		cmp -s "$replacement_file" "$source/$replacement_path" || cp "$replacement_file" "$source/$replacement_path"
	fi
	write_source_file_list "$source" "$current_files"

	awk '{sub(/^[^ ]+  /, ""); print}' "$manifest_file" | LC_ALL=C sort >"$expected_files"
	cmp -s "$current_files" "$expected_files" || {
		printf '%s source does not exactly match the tested JetPack backport.\n' "$name" >&2
		rm -f "$current_files" "$expected_files"
		exit 1
	}
	(cd "$source" && sha256sum -c "$manifest_file")
	rm -f "$current_files" "$expected_files"
}

ensure_patched_source horde-engine "$engine_repository" "$engine_source" "$engine_commit" \
	"$engine_patch" "$engine_manifest" "$engine_base_replacement" hordelib/model_manager/base.py
ensure_patched_source horde-model-reference "$model_reference_repository" "$model_reference_source" \
	"$model_reference_commit" "$model_reference_patch" "$model_reference_manifest"
ensure_patched_source horde-sdk "$sdk_repository" "$sdk_source" "$sdk_commit" "$sdk_patch" "$sdk_manifest"

SETUPTOOLS_SCM_PRETEND_VERSION=3.0.0 \
	SETUPTOOLS_SCM_PRETEND_VERSION_FOR_HORDE_ENGINE=3.0.0 \
	"$python" -m pip install --no-deps "$engine_source"
SETUPTOOLS_SCM_PRETEND_VERSION=5.1.1 \
	SETUPTOOLS_SCM_PRETEND_VERSION_FOR_HORDE_MODEL_REFERENCE=5.1.1 \
	"$python" -m pip install --no-deps "$model_reference_source"
SETUPTOOLS_SCM_PRETEND_VERSION=0.20.7 \
	SETUPTOOLS_SCM_PRETEND_VERSION_FOR_HORDE_SDK=0.20.7 \
	"$python" -m pip install --no-deps "$sdk_source"
"$python" -m pip install --no-deps \
	"$triton_wheel" "$comfy_kitchen_wheel" "$flash_attention_wheel" "$xformers_wheel"
"$python" -m pip install --no-deps -e "$script_dir"

"$python" -m pip check

"$python" -s - <<'PY'
import importlib.metadata as metadata

import horde_model_reference
import horde_safety
import horde_sdk
import hordelib
import comfy_kitchen
import flash_attn
import torch
import textual
import textual_serve
import triton
import xformers

assert torch.__version__ == "2.1.0a0+git7bcf7da", torch.__version__
assert torch.version.cuda == "11.4", torch.version.cuda
assert torch.cuda.is_available(), "PyTorch cannot access Xavier's CUDA device"
assert torch.cuda.get_device_capability(0) == (7, 2)
assert xformers._has_cpp_library, "xFormers CUDA extension is unavailable"
assert metadata.version("horde-worker-regen") == "12.0.0"
assert metadata.version("horde-engine") == "3.0.0"
assert metadata.version("horde-sdk") == "0.20.7"
assert metadata.version("horde-model-reference") == "5.1.1"
assert metadata.version("triton") == "2.1.0+xavierjp5"
assert metadata.version("comfy-kitchen") == "0.2.26"
assert metadata.version("flash-attn-legacy") == "0.5.0+xavierjp5"
assert metadata.version("haidra-core") == "0.0.5"
assert metadata.version("textual") == "8.1.1"

backends = comfy_kitchen.list_backends()
assert backends["eager"]["available"], backends
assert backends["triton"]["available"], backends
for device in ("cpu", "cuda"):
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32, device=device)
    qweight = torch.tensor([[0x98, 0xBA]], dtype=torch.uint8, device=device).view(torch.int8)
    wscales = torch.ones((1, 1), dtype=torch.float32, device=device)
    wzeros = torch.zeros((1, 1), dtype=torch.float32, device=device)
    result = comfy_kitchen.gemv_awq_w4a16(x, qweight, wscales, wzeros, group_size=4)
    torch.testing.assert_close(result.cpu(), torch.tensor([[20.0]]))

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
print("xformers", metadata.version("xformers"))
print("worker", metadata.version("horde-worker-regen"))
print("engine", metadata.version("horde-engine"))
print("sdk", metadata.version("horde-sdk"))
print("model reference", metadata.version("horde-model-reference"))
print("triton", triton.__version__)
print("comfy kitchen", metadata.version("comfy-kitchen"))
print("flash attention", metadata.version("flash-attn-legacy"), flash_attn.__name__)
print("textual", metadata.version("textual"))
print("horde imports", hordelib.__name__, horde_sdk.__name__, horde_safety.__name__, horde_model_reference.__name__)
PY
