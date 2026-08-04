# JetPack 5 Xavier reGen 10.1.2.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit, validated JetPack 5 / CUDA 11.4 runtime to
Tazlin's tagged reGen `v10.1.2.1` release while leaving all generic platform
paths unchanged.

**Architecture:** Keep JetPack support in dedicated POSIX shell scripts and a
dedicated requirements overlay. Port the proven Xavier runtime fixes into the
existing 10.1.2.1 classes with regression tests, then install and smoke-test
the branch in an isolated Xavier directory and virtual environment.

**Tech Stack:** Python 3.10.20, pytest, Pydantic 2, POSIX `sh`, NVIDIA
JetPack 5 / L4T R35.6.4, CUDA 11.4, NVIDIA PyTorch 2.1.0a0, xFormers 0.0.23,
and Tazlin horde-engine source.

---

## Task 1: Add JetPack Packaging Contracts

**Files:**

- Create: `tests/test_jetson_packaging.py`
- Create: `requirements.jetson-jp5.txt`
- Create: `install-jetson-jp5.sh`
- Create: `start-jetson-jp5.sh`
- Create: `build-xformers-jetson-jp5.sh`
- Modify: `README.md`

- [ ] **Step 1: Write failing packaging tests**

Add tests that read the shipped files and require:

```python
def test_jetson_installer_pins_xavier_runtime() -> None:
    installer = (ROOT / "install-jetson-jp5.sh").read_text()
    requirements = (ROOT / "requirements.jetson-jp5.txt").read_text()

    assert "3.10.20" in installer
    assert "torch-2.1.0a0+git7bcf7da" in installer
    assert "torchvision-0.16.0+fbb4cc5" in installer
    assert "torchaudio-2.1.0+6ea1133" in installer
    assert "a0555b474696257a2374f4d1d4bc10b3d3fae5e3" in installer
    assert "xformers-0.0.23+e1b36f7.d*" in installer
    assert "onnxruntime==1.17.3" in requirements
    assert "horde_sdk~=0.17.1" in requirements
    assert "CMAKE_BUILD_PARALLEL_LEVEL=1" in installer
    assert "MAX_JOBS=1" in installer
```

Add separate tests for the launcher CPU calculation, xFormers build isolation,
architecture/L4T checks, and installed package validation.

- [ ] **Step 2: Verify packaging tests fail**

Run on the Xavier with one test process:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    .venv/bin/python -m pytest -q tests/test_jetson_packaging.py
```

Expected: failures because the JetPack files do not exist.

- [ ] **Step 3: Add the dedicated JetPack requirements overlay**

Create `requirements.jetson-jp5.txt` without a `torch` dependency. Pin the
worker-facing Horde dependencies from reGen 10.1.2.1, retain the known-good
aarch64 media/runtime versions, and pin `onnxruntime==1.17.3` because newer
aarch64 builds abort on Xavier CPU detection.

- [ ] **Step 4: Add the installer**

Implement `install-jetson-jp5.sh` with `set -eu`. It must:

```sh
test "$(uname -m)" = aarch64
grep -q '^# R35 ' /etc/nv_tegra_release
version=$("$python_bin" -c \
    'import sys; print(".".join(map(str, sys.version_info[:3])))')
test "$version" = 3.10.20
export CMAKE_BUILD_PARALLEL_LEVEL=1
export MAX_JOBS=1
```

Validate SHA-256 sidecars for the three NVIDIA wheels and the local xFormers
wheel, create `.venv`, install the NVIDIA wheels with `--no-deps`, install the
JetPack overlay, install Tazlin's horde-engine source at commit
`a0555b474696257a2374f4d1d4bc10b3d3fae5e3` with `--no-deps`, install the
worker editable with `--no-deps`, and run import/version assertions.

- [ ] **Step 5: Add launcher and xFormers builder**

Use strict POSIX `sh`. The launcher loads `.env`, selects CUDA device 0, uses
half the configured CPUs for runtime libraries, preloads `libGLdispatch`, runs
model download, then `exec`s the worker. The builder pins xFormers commit
`e1b36f781ba1c9d10f36fc0ec87170e0b381fdad`, compute capability `7.2`, and
every native build control to one thread.

- [ ] **Step 6: Verify and commit packaging**

Run the focused test, `shfmt -d`, `shellcheck`, and `git diff --check`.
Commit as `feat: add JetPack 5 runtime for reGen 10`.

## Task 2: Preserve Deployment Secrets and Slow-Worker Configuration

**Files:**

- Modify: `tests/test_bridge_data.py`
- Modify: `horde_worker_regen/bridge_data/data_model.py`
- Modify: `bridgeData_template.yaml`

- [ ] **Step 1: Write failing configuration tests**

Add tests proving:

```python
monkeypatch.setenv("AIWORKER_API_KEY", "test-key")
bridge_data = reGenBridgeData.model_validate({})
bridge_data.load_env_vars()
assert bridge_data.api_key == "test-key"
```

```python
monkeypatch.setenv("AIWORKER_CIVITAI_API_TOKEN", "test-token")
bridge_data.load_env_vars()
assert bridge_data.CIVIT_API_TOKEN == "test-token"
assert os.environ["CIVIT_API_TOKEN"] == "test-token"
```

Also assert that `extra_slow_worker=True` preserves an explicitly configured
`post_process_job_overlap=False` and raises `preload_timeout` to at least 120.

- [ ] **Step 2: Verify the new tests fail for missing environment mapping**

Run `pytest -q tests/test_bridge_data.py -k 'environment or extra_slow'`.

- [ ] **Step 3: Implement minimal environment mapping**

At the start of `load_env_vars()`, apply `AIWORKER_API_KEY` and
`AIWORKER_CIVITAI_API_TOKEN` only when present. Set both the model field and the
engine-facing `CIVIT_API_TOKEN` environment variable.

- [ ] **Step 4: Keep explicit overlap configuration authoritative**

Do not force `post_process_job_overlap=True` in extra-slow mode. Keep the
existing queue, thread, memory-mode, and preload timeout constraints.

- [ ] **Step 5: Add conservative JetPack template guidance**

Document a one-thread, queue-zero, extra-slow, safety-on-GPU profile with post
processing overlap disabled. Do not globally enable or disable ControlNet.

- [ ] **Step 6: Verify and commit configuration behavior**

Run `pytest -q tests/test_bridge_data.py tests/test_jetson_packaging.py` and
scoped Ruff checks. Commit as `fix: support Xavier deployment configuration`.

## Task 3: Harden Child Process State and IPC Recovery

**Files:**

- Create: `tests/test_jetson_process_recovery.py`
- Modify: `horde_worker_regen/process_management/process_manager.py`

- [ ] **Step 1: Write failing process-state tests**

Add focused tests for:

- `BrokenPipeError` logs a debug recovery message and returns `False`.
- Unexpected send failures remain error logs.
- `is_process_alive()` returns true for a live non-ending process.
- Available inference count excludes safety and starting processes.

Construct `HordeProcessInfo` with the required `process_launch_identifier` and
use real enum states so the tests exercise production predicates.

- [ ] **Step 2: Verify the four tests fail**

Run:

```sh
pytest -q tests/test_jetson_process_recovery.py \
    -k 'broken_pipe or running_process or available_inference'
```

- [ ] **Step 3: Correct process predicates and broken-pipe handling**

Implement:

```python
return self.mp_process.is_alive() and self.last_process_state not in {
    HordeProcessState.PROCESS_ENDING,
    HordeProcessState.PROCESS_ENDED,
}
```

Count availability only when `process_type` is inference and
`can_accept_job()` is true. Catch `(BrokenPipeError, EOFError, OSError)` before
the generic exception and log that the control channel is already closed.

- [ ] **Step 4: Verify and commit IPC recovery**

Run the focused tests plus existing process-manager tests, then commit as
`fix: harden Xavier child process recovery`.

## Task 4: Make Model-Change Cycling Lossless

**Files:**

- Modify: `tests/test_jetson_process_recovery.py`
- Modify: `horde_worker_regen/process_management/process_manager.py`

- [ ] **Step 1: Write failing model-cycle tests**

Test that a waiting inference process holding another model is deliberately
cycled before preload, and that the deliberate cycle calls:

```python
manager._replace_inference_process(process, fault_referenced_job=False)
```

Assert the replacement order is end, drain pending process messages, then
start, and that the referenced completed job is not faulted.

- [ ] **Step 2: Verify the model-cycle tests fail**

Run:

```sh
pytest -q tests/test_jetson_process_recovery.py \
    -k 'model_switch or intentional_process_cycle'
```

- [ ] **Step 3: Add an intentional-cycle mode**

Add keyword-only `fault_referenced_job: bool = True` to
`_replace_inference_process`. Skip job faulting for deliberate model changes,
drain messages between ending and restarting, and return a truthy scheduling
result after initiating the cycle so the manager does not treat it as an idle
preload pass.

- [ ] **Step 4: Verify and commit model cycling**

Run all process recovery tests and commit as
`fix: preserve jobs during Xavier model cycles`.

## Task 5: Serialize Safety Work and Extend Slow R2 Uploads

**Files:**

- Modify: `tests/test_jetson_process_recovery.py`
- Modify: `horde_worker_regen/process_management/process_manager.py`
- Modify: `horde_worker_regen/process_management/safety_process.py`

- [ ] **Step 1: Write failing safety and upload tests**

Test that safety dispatch marks the process state and control flag before the
job leaves the pending list. Test that `get_r2_upload_timeout_seconds()` returns
60 for extra-slow workers and 10 otherwise.

- [ ] **Step 2: Verify both tests fail**

Run:

```sh
pytest -q tests/test_jetson_process_recovery.py \
    -k 'safety_dispatch or r2_upload'
```

- [ ] **Step 3: Implement safety reservation and timeout selection**

After a safety message is accepted, set the process to
`EVALUATING_SAFETY` and `EVALUATE_SAFETY` before another scheduling pass can
select it. Add a timeout helper and use its value for both aiohttp's total
timeout and the enclosing `asyncio.wait_for` timeout.

- [ ] **Step 4: Keep safety heartbeats valid during slow checks**

Ensure the safety process emits or updates heartbeat/state information during
long checks so the watchdog does not replace a healthy process solely because
Xavier safety evaluation is slow.

- [ ] **Step 5: Verify and commit slow-path stability**

Run all focused bridge, process, and packaging tests. Commit as
`fix: stabilize Xavier safety and uploads`.

## Task 6: Validate the Isolated 10.x Runtime on Xavier

**Files:**

- Modify: `README.md` only if validation exposes missing operator guidance

- [ ] **Step 1: Sync the branch to an isolated Xavier build directory**

Use `/mnt/xavier-ssd/build/horde-worker-v10.1.2.1-xavier`. Do not write into
`/media/jack/SDCard/horde-worker-reGen-xavier-jp5`.

- [ ] **Step 2: Run the complete focused test set**

Run bridge, packaging, process recovery, dependency, and version tests with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`.

- [ ] **Step 3: Install with exact one-thread controls**

Run the JetPack installer with `CMAKE_BUILD_PARALLEL_LEVEL=1`, `MAX_JOBS=1`,
and the existing verified wheel directory. Do not launch the game or worker
for live job processing.

- [ ] **Step 4: Validate runtime identity**

Assert Python 3.10.20, torch 2.1.0a0 CUDA 11.4, Xavier capability 7.2,
xFormers compiled operators, worker 10.1.2, and successful imports of the
Tazlin horde-engine source, SDK, safety, and model-reference packages.

- [ ] **Step 5: Run non-live startup checks**

Parse the production-compatible config from a copy, resolve the configured
models against the NVMe cache, and run model download verification without
starting the production worker or accepting Horde jobs.

- [ ] **Step 6: Run final quality gates**

Run focused pytest, Ruff dry-run then Ruff, `shfmt -d`, `shellcheck`, and
`git diff --check`. Review the branch diff for only JetPack-required changes.

- [ ] **Step 7: Commit and push**

Commit any validation documentation, verify the branch is ahead of
`v10.1.2.1`, then push `xavier-jp5-cuda114-v10.1.2.1-tazlin` to the user's
fork. Do not open or merge a pull request unless requested.
