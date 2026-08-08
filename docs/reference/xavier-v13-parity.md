# Xavier v13 parity

This page tracks the JetPack 5 compatibility port from the deployed Xavier
worker to the final v13 worker line. It separates source-level parity from
runtime feasibility so an upstream feature is not marked complete merely
because its Python code imports on a desktop machine.

## Baselines

| Role | Revision | Notes |
| --- | --- | --- |
| Deployed Xavier | worker `12.0.0` + JP5 patches | Custom aarch64 stack |
| First v13 API | `v13.0.0` (`398c2cc6`) | New process architecture |
| Parity target | `v13.16.7` (`53ee398b`) | Final v13 release before v14 |
| Audited dev tree | `v17.10.0` (`2b04c0c`) | Not the first port target |

The v12.0.0 to v13.0.0 transition changes 698 files with roughly 85,000
insertions and 10,000 deletions. The remainder of the v13 line adds another 419
changed files. Porting individual old filenames is therefore the wrong model:
many process-management modules moved into lifecycle, resources, scheduling,
models, workers, and simulation packages.

## Hard compatibility boundaries

- **Python:** v13 expects 3.12. The compatibility runtime is Python 3.10.20.
  Maintain a narrow patch set and test imports continuously.
- **PyTorch:** v13 locks current desktop CUDA builds. Xavier uses NVIDIA's
  aarch64 PyTorch 2.1 CUDA 11.4 wheel. Backport API use instead of upgrading the
  JetPack CUDA stack.
- **horde-engine:** v13.16.7 requires `>=3.11.0,<4`. Rebase the existing Python
  3.10 and JetPack patch onto the required v3 revision.
- **horde-model-reference:** v13 requires `>=7.0.2,<8`, while production uses a
  patched 5.1.1. Port the compatibility patch to v7 and verify cache behavior.
- **horde-sdk:** v13 requires `~=0.22.4`, while production uses a patched 0.20.7.
  Port the Python 3.10 compatibility and API changes to 0.22.x.
- **CUDA bootstrap:** v13 offers desktop `cu126`, `cu130`, and `cu132` wheels.
  Preserve the JetPack installer and never auto-select desktop `cu126`.
- **Attention:** Xavier uses custom `sm_72` xFormers and legacy FlashAttention.
  Keep xFormers in production and FlashAttention compatibility-only.
- **Telemetry:** MediaPipe 0.10.18 requires protobuf 4. Keep Logfire at 4.6.0,
  OpenTelemetry at 1.27.0 / instrumentation 0.48b0, and importlib-metadata at
  8.4.0. Newer OpenTelemetry protocol releases require protobuf 5 and cannot
  coexist with the Xavier controlnet stack.
- **OpenMP:** PyTorch loads JetPack's system `libgomp`. The PyPI scikit-learn
  wheel bundles a renamed second copy that can exhaust glibc's static TLS block
  during the Horde Safety import chain. Use the checksum-pinned Xavier wheel
  repaired to link `libgomp.so.1` instead.
- **FLUX decode:** The standalone controlled-release probe demonstrates one
  lower-retention execution shape, but it is not part of the worker runtime.
  Before advertising FLUX, the normal v13 worker path must generate an image,
  pass Horde safety, and submit it successfully under operator control.

## Port sequence

### P0: preserve the known-good production envelope

- [x] Record the Xavier runtime and wheel contract.
- [x] Record unified-memory and NvMap behavior.
- [x] Keep FLUX excluded from production advertisement.
- [x] Preserve xFormers as the production backend.
- [x] Add a repeatable staged FLUX probe that remains outside the runtime.

### P1: make the v13 tree buildable without pretending it is runnable

- [x] Port the JetPack installer, wheel checksum manifest, and single-threaded
  xFormers builder onto `v13.16.7`.
- [x] Detect JetPack 5 before desktop NVIDIA and refuse generic CUDA wheels.
- [x] Replace v12 hard-coded package assertions with v13 dependency pins.
- [x] Rebase the horde-engine, horde-model-reference, and horde-sdk Python 3.10
  patches onto the v13-required revisions.
- [x] Add static packaging tests that reject desktop CUDA wheels and parallel
  compiler settings on Xavier.
- [x] Resolve the complete Python 3.10 aarch64 wheel graph without desktop CUDA
  substitutions or protobuf conflicts.
- [x] Produce a clean Python 3.10 import and `pip check` result.

The dependency gate was validated on August 8, 2026 on the Xavier running L4T
R35.6.4. The clean r3 environment at
`/mnt/xavier-ssd/build/venvs/horde-worker-v13.16.7-jp5-deps-20260808-r3`
reported no broken requirements and imported worker `13.16.7`, Horde Engine
`3.11.0`, Horde SDK `0.22.4`, model reference `7.0.2`, Horde Safety, PyTorch,
xFormers, Triton, Comfy Kitchen, and scikit-learn in the production import
order. All 18 scikit-learn OpenMP extensions linked `libgomp.so.1`, with no
private `libgomp` dependency. CPU and CUDA Comfy Kitchen probes passed while
the deployed v12 worker remained running. The repaired scikit-learn wheel has
SHA-256 `85d2ee85b87bebdcad8e168a8cf327a77931583bc661dd5728f26edff2413074`.
The complete device report is stored at
`/mnt/xavier-ssd/build/horde-worker-v13.16.7-jp5-deps-20260808/dependency-validation-r3.txt`.

### P2: recover worker behavior

- [x] Port the production RAM floor, pop hold, idle-child recycling, and bounded
  OOM recovery onto the v13 process-management packages.
- [x] Verify both directions of supervisor and child-process shutdown/restart.
- [x] Verify model download, safety, inference, and submission processes
  independently before enabling network job pops.
- [x] Run the v13 dry-run and fake-worker suites with exactly one compiler thread.

The P2 gate was validated on August 8, 2026 with Python 3.10.20 on the physical
Xavier, without enabling network job pops. The focused RAM-pressure, bounded
OOM recovery, process lifecycle, supervisor command, model download, safety,
inference, submission, dry-run, and fake-worker suites passed 543 tests. The
run used single-thread build environment controls and is recorded at
`/mnt/xavier-ssd/build/horde-worker-v13.16.7-jp5-tests-20260808/p2-verification.log`.

The device run found and fixed three remaining compatibility and recovery
edges: a Python 3.12-only generic helper in the fake-worker tests, a Linux
aarch64 child-spawn timeout that was too tight for JetPack 5, and a stale RAM
drain marker that could keep job pops held after the process became idle and
the host recovered. Direct regressions now cover the popper's RAM hold,
drain-release behavior, supervisor-requested process replacement, and
supervisor-requested graceful shutdown.

### P3: restore accelerator capabilities

- [x] Re-run the exact FLUX head-dimension-128 xFormers and FlashAttention
  compatibility probes.
- [ ] Integrate staged FLUX component ownership into the normal pipeline.
- [ ] Require 1024x1024 tiled VAE completion with measured release at both stage
  boundaries.
- [ ] Soak supported SDXL jobs, LoRAs, img2img, safety, and post-processing.
- [ ] Advertise FLUX only after repeated network jobs submit without NvMap or
  allocator corruption.

The head-dimension-128 compatibility gate was repeated on August 8, 2026 with
the rebuilt `flash-attn-legacy==0.5.1+xavierjp5fa2` wheel. The offline probe used
shape `(1, 256, 2, 128)` and three synchronized iterations per backend. xFormers
and legacy FlashAttention both matched PyTorch SDPA with maximum absolute error
`1.52587890625e-05`; xFormers averaged 0.776 ms and legacy FlashAttention
averaged 1.024 ms. Available system RAM remained 19,531,488 KiB before and after
the probe. The report SHA-256 is
`1e6d96c3a997bef0ef65f5499fa308fe5cc5e01d28a0fbc2b7c62272e521af0e` and the
device copy is stored at
`/mnt/xavier-ssd/build/horde-worker-v13.16.7-jp5-attention-probe-20260808/attention-compat-head128.json`.
The deployed v12 worker remained running, and no v13 worker or network job pop
was started.

## Acceptance evidence

A parity item is complete only when its evidence is saved with:

- exact worker and dependency revisions,
- wheel names and SHA-256 checksums,
- Python, CUDA, L4T, and compute-capability values,
- before/after available RAM and process RSS at each heavy stage,
- child exit status and kernel/NvMap messages,
- attention backend and observed head dimension,
- generated artifact hash when inference is involved, and
- production restoration status after an offline test.

Passing on a desktop CUDA runner is useful but does not satisfy a Xavier parity
gate. Conversely, a locally patched run is not complete until the patch and its
test are committed to this branch.
