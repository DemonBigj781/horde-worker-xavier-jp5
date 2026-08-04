# JetPack 5 Xavier Support for reGen 10.1.2.x

## Goal

Port the validated NVIDIA Jetson AGX Xavier runtime from the 9.0.7 branch to
the `10.1.2` point in Tazlin's fork lineage, then advance through compatible
upstream commits without changing the normal CUDA, ROCm, DirectML, or CPU
installation paths.

The result is an explicit legacy install profile for JetPack 5 / L4T R35.6.4,
CUDA 11.4, aarch64, and Python 3.10.20. It must be testable beside the live
9.0.7 worker and must not replace that worker until on-device validation is
complete.

## Baseline

- Source lineage: `tazlin/horde-worker-reGen` at commit `3963ae36`, which
  corresponds to the upstream `v10.1.2.1` tag.
- Worker package version: `10.1.2` in both package version sources.
- Target: Jetson AGX Xavier 32 GB, compute capability 7.2.
- Platform: JetPack 5 / L4T R35.6.4, CUDA 11.4, aarch64.
- Python: exactly 3.10.20.
- PyTorch: NVIDIA aarch64 wheel `2.1.0a0+git7bcf7da`, CUDA 11.4.
- xFormers: locally built `0.0.23+e1b36f7.d20260803`, CUDA extension enabled.

The untagged `404af300` commit is intentionally excluded. Although its
`pyproject.toml` still says `10.1.2`, its package reports `12.0.0` and requires
Python 3.12-era dependencies, so it is not a steady 10.x upgrade boundary.

### Incremental Upgrade Ladder

The next compatible source milestone is `92cb5ddf`, which moves dependency
management and launch tooling to `uv` while retaining worker version `10.1.2`,
Python `>=3.10,<3.13`, `horde_sdk~=0.17.1`,
`horde_model_reference~=0.9.2`, and `horde_engine~=2.20.12`. This milestone is
the basis of the `xavier-jp5-cuda114-v10.1.2.2-tazlin` branch.

Later commits remain eligible one at a time until `404af300`. The published
`v12.0.0` boundary is a separate compatibility project because it raises the
worker, Horde SDK, and model-reference runtime to Python 3.12 while NVIDIA's
JetPack 5 CUDA 11.4 wheels remain tied to CPython 3.10.

### Canonical Source Map

- `google-ai-edge/mediapipe`: MediaPipe source; JetPack remains pinned to
  `0.10.18`, the available Python 3.10 aarch64 wheel.
- `tazlin/hordelib`: image engine source, pinned to commit `a0555b4` for the
  10.1.2.x runtime.
- `tazlin/horde-model-reference`: requested fork lineage; its current `main`
  is older than the modern Haidra releases, so every use must name an exact
  compatible commit instead of following `main`.
- `Haidra-Org/horde-safety`: safety library source; 10.1.2.x uses `0.3.0`.
- `Haidra-Org/ComfyUI`: engine-integrated ComfyUI source, not an independent
  worker install target.
- `Haidra-Org/haidra-core`: a dependency of newer model-reference releases;
  it is not required by the 10.1.2.x `horde_model_reference~=0.9.2` line.
- `Haidra-Org/AI-Horde-text-model-reference` and
  `Haidra-Org/AI-Horde-CLI`: adjacent text-model data and client tooling, not
  dependencies of the image worker runtime.

## Architecture

### Explicit Legacy Profile

Add JetPack-specific install, launch, and xFormers build scripts. The generic
runtime scripts remain unchanged. The profile installs exact NVIDIA aarch64
wheels and rejects desktop CUDA wheels, generic cu118 wheels, wrong Python
versions, wrong architectures, and mismatched xFormers builds.

### Dependency Isolation

The JetPack profile uses its own `.venv` and a dedicated requirements overlay.
It installs the tested PyTorch family first with `--no-deps`, installs the
worker and compatible Horde dependencies without allowing pip to replace the
NVIDIA torch wheel, and validates the installed runtime before launch.

Tazlin's `hordelib` fork may be used only at an explicitly pinned compatible
commit if reGen 10.1.2.1 requires it. `tazlin/hordebridge` is not part of this
image-worker runtime because it is a separate text-generation bridge.

### Runtime Stability

Port only behavior proven necessary on the Xavier 9.0.7 worker:

- API and Civitai secrets load from environment variables.
- Broken IPC pipes cause bounded process recovery instead of worker death.
- Intentional model-change cycles preserve completed jobs and drain messages.
- Process availability counts exclude processes being restarted.
- Safety dispatch marks the safety process busy before another job can use it.
- Extra-slow mode preserves an explicitly disabled post-process overlap.
- Extra-slow workers receive the longer R2 upload timeout.
- The JetPack template uses one inference thread, no overlap, and conservative
  model cycling suitable for shared system/GPU memory.

The port follows the 10.1.2.1 code structure instead of copying entire 9.0.7
modules over newer files.

## Verification

1. Add packaging and behavior regression tests before production changes.
2. Confirm each new test fails for the missing JetPack behavior.
3. Run focused tests locally using Python 3.10-compatible dependencies.
4. Run the same tests on the Xavier with one compiler/test worker.
5. Build/install into a separate test directory and virtual environment.
6. Validate exact Python, torch, CUDA, GPU capability, xFormers, worker, and
   Horde dependency versions.
7. Run model download/config parsing and a non-live startup smoke test.
8. Do not launch or replace the production worker as part of this branch.

## Acceptance Criteria

- The branch is based on commit `3963ae36` in Tazlin's fork lineage and may
  include only individually validated pre-`404af300` commits.
- Generic platform scripts and dependency resolution remain unchanged.
- JetPack install fails clearly for an unsupported architecture or version.
- All native build controls use exactly one CPU thread.
- The installed runtime retains NVIDIA's CUDA 11.4 PyTorch wheel.
- xFormers reports its compiled CUDA extension and Xavier-compatible operators.
- Focused unit, packaging, lint, and on-device tests pass.
- The branch is committed and pushed to the user's fork only after validation.

## Exclusions

- No upgrade beyond the 10.x release line in this branch.
- No deployment over the live 9.0.7 worker.
- No generic cu118 substitution for NVIDIA's JetPack CUDA 11.4 torch wheel.
- No ControlNet policy changes; those remain deployment configuration choices.
- No integration of the unrelated `tazlin/hordebridge` text worker.
