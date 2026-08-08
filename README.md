# Horde Worker v13 - Xavier JetPack 5

This is the main integration repository for running Horde Worker v13 on NVIDIA Jetson AGX Xavier. The target is JetPack 5, Ubuntu 20.04, Python 3.10, CUDA 11.4, and Xavier's shared system/GPU memory.

## Current state

- The Xavier dependency environment installs and passes `pip check`.
- A focused physical-device process-management suite has passed 543 tests.
- The tested xFormers and legacy Flash Attention paths agree numerically in the compatibility probe.
- The Python 3.10 compatibility port is source tracked.
- Full network-connected v13 production operation has not been proven.
- FLUX and large SDXL jobs still require end-to-end component-lifetime validation through the normal worker path.
- The controlled-release FLUX probe is a standalone diagnostic, not a runtime handler. It remains outside the worker until an operator-run v13 image generates, passes Horde safety, and submits successfully.

## Safety boundary

This port does not bypass the Horde safety process or replace the bridge protocol. Image pop requests identify this fork as `AI Horde Worker Xavier JP5` and point to this repository. Model loading, inference, safety inspection, and submission must remain compatible with the established worker contract. Automated development checks must not launch a production worker or submit network jobs; the operator performs physical end-to-end runs manually.

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
