# Horde Worker v13 - Xavier JetPack 5

This is the main integration repository for running Horde Worker v13 on NVIDIA Jetson AGX Xavier. The target is JetPack 5, Ubuntu 20.04, Python 3.10, CUDA 11.4, and Xavier's shared system/GPU memory.

## Current state

- The Xavier dependency environment installs and passes `pip check`.
- The exact upstream `v13.16.7` runtime surface is frozen across 242 Python
  files and checked for missing modules, definitions, or signature drift.
- The physical Xavier collects all 3,049 tests under Python 3.10. The complete
  offline non-GPU gate passes 3,013 tests with 12 existing skips; the remaining
  24 tests are real-GPU capability checks reserved for the operator-controlled
  image trial.
- The tested xFormers and legacy Flash Attention paths agree numerically in the compatibility probe.
- The Python 3.10 compatibility port is source tracked.
- Full network-connected v13 production operation has not been proven.
- FLUX remains source-present for upstream parity but is excluded from the v13
  deployment path. It will be reconsidered in a later worker version after the
  newer memory-accounting work.
- The controlled-release FLUX probe is a standalone diagnostic, not a runtime
  handler. The alternative lifecycle handler remains outside the worker.

## Safety boundary

This port does not bypass the Horde safety process or replace the bridge protocol. Image pop requests identify this fork as `AI Horde Worker Xavier JP5` and point to this repository. Model loading, inference, safety inspection, and submission must remain compatible with the established worker contract. Automated development checks must not launch a production worker or submit network jobs; the operator performs physical end-to-end runs manually. The existing trial checkout remains frozen at its pre-parity commit and is not authorized for launch until it is rebuilt from the completed parity branch.

## Release gates

A v17 release requires a continuous 24-hour physical Xavier session with zero worker, child-process, supervisor, or service recovery events and no downtime. Aggregate active-work throughput must remain above 0.15 MPS/s. Brief dips for larger SDXL or FLUX jobs are acceptable only when the full-session average remains above the threshold.

## Start here

- `docs/how-to/run-on-jetson-xavier.md`
- `docs/reference/xavier-v13-parity.md`
- `requirements.jetson-jp5.txt`
- `install-jetson-jp5.sh`
- `start-jetson-jp5.sh`

## Upstream

Forked from `tazlin/horde-worker-reGen`. General worker documentation remains available upstream; this repository is the Xavier-specific development and validation line.
