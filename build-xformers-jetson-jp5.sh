#!/bin/sh
set -eu

build_root=${XFORMERS_BUILD_ROOT:-/mnt/xavier-ssd/build}
source_dir=$build_root/xformers-e1b36f7
wheel_dir=${JETSON_WHEEL_DIR:-${HOME}/jetson}
wheel_torch=$wheel_dir/torch-2.1.0a0+git7bcf7da-cp310-cp310-linux_aarch64.whl
python_bin=${PYTHON_BIN:-${HOME}/.pyenv/versions/3.10.20/bin/python}
venv_dir=${XFORMERS_BUILD_VENV:-$build_root/xformers-build-venv}

test "$(uname -m)" = aarch64 || {
	printf '%s\n' 'xFormers must be built on the aarch64 Xavier target.' >&2
	exit 1
}
test -f "$wheel_torch" || {
	printf 'Required Jetson PyTorch wheel not found: %s\n' "$wheel_torch" >&2
	exit 1
}
test ! -e "$source_dir" || {
	printf 'Refusing to overwrite existing build directory: %s\n' "$source_dir" >&2
	exit 1
}
test ! -e "$venv_dir" || {
	printf 'Refusing to overwrite existing build environment: %s\n' "$venv_dir" >&2
	exit 1
}

"$python_bin" -m venv "$venv_dir"
python=$venv_dir/bin/python
"$python" -m pip install --upgrade pip "setuptools==80.9.0" wheel ninja numpy==1.26.4
"$python" -m pip install --no-deps "$wheel_torch"

git clone --recursive https://github.com/facebookresearch/xformers.git "$source_dir"
git -C "$source_dir" switch --detach e1b36f781ba1c9d10f36fc0ec87170e0b381fdad
git -C "$source_dir" submodule update --init --recursive

python3 - "$source_dir/setup.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = '                "--threads",\n                "4",\n'
new = '                "--threads",\n                "1",\n'
if old not in text:
    raise SystemExit("xFormers NVCC thread block was not found")
path.write_text(text.replace(old, new, 1))
PY

mkdir -p "$wheel_dir"
cd "$source_dir"
export CUDA_HOME=/usr/local/cuda-11.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST=7.2
export XFORMERS_DISABLE_FLASH_ATTN=1
export XFORMERS_BUILD_TYPE=Release
export FORCE_CUDA=1
export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export CARGO_BUILD_JOBS=1
export MAKEFLAGS=-j1
export NVCC_THREADS=1
export NINJAFLAGS=-j1

"$python" setup.py bdist_wheel --dist-dir "$wheel_dir"
set -- "$wheel_dir"/xformers-0.0.23+e1b36f7.d*-cp310-cp310-linux_aarch64.whl
test "$#" -eq 1 && test -f "$1" || {
	printf '%s\n' 'Expected exactly one built xFormers wheel.' >&2
	exit 1
}
sha256sum "$1" >"$1.sha256"
