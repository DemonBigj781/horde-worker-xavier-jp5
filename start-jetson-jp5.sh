#!/bin/sh
set -eu

script_dir=$(
	CDPATH=
	cd -P "$(dirname "$0")"
	pwd
)
cd "$script_dir"

if test -f "$script_dir/.env"; then
	set -a
	# shellcheck disable=SC1090,SC1091
	. "$script_dir/.env"
	set +a
fi

export CUDA_VISIBLE_DEVICES=0
configured_cpus=$(getconf _NPROCESSORS_CONF)
compute_threads=$((configured_cpus / 2))
if test "$compute_threads" -lt 1; then
	compute_threads=1
fi
export OMP_NUM_THREADS=$compute_threads
export OPENBLAS_NUM_THREADS=$compute_threads
export NUMEXPR_NUM_THREADS=$compute_threads
export LD_PRELOAD="/lib/aarch64-linux-gnu/libGLdispatch.so.0${LD_PRELOAD:+:$LD_PRELOAD}"

"$script_dir/.venv/bin/python" -s "$script_dir/download_models.py"
exec "$script_dir/.venv/bin/python" -s "$script_dir/run_worker.py" "$@"
