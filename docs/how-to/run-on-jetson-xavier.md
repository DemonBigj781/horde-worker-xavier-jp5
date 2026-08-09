# Running on Jetson AGX Xavier

Jetson AGX Xavier support is an explicit compatibility port, not a normal CUDA
installation. JetPack 5 supplies CUDA 11.4 and Python 3.10 on an aarch64 system,
while the standard v13 worker expects Python 3.12 and current desktop PyTorch
CUDA wheels. Do not let the generic bootstrap select `cu126`: those wheels are
not a valid JetPack 5 runtime.

## Supported runtime contract

The validated development system uses:

| Component | Xavier compatibility runtime |
| --- | --- |
| Device | Jetson AGX Xavier, compute capability 7.2 |
| OS stack | JetPack 5 / L4T R35.6.4 |
| Python | 3.10.20 |
| CUDA | 11.4 |
| PyTorch | `2.1.0a0+git7bcf7da` aarch64 wheel |
| torchvision | `0.16.0+fbb4cc5` aarch64 wheel |
| xFormers | Xavier-specific `0.0.23` build for `sm_72` |
| FlashAttention | Legacy Xavier build, compatibility checks only |

Production inference uses xFormers. FlashAttention must remain installed and
testable as a compatibility layer, but it is not the preferred production
backend. An exact FLUX-shaped direct test with Q/K/V shape
`(1, 4352, 24, 128)` completed with finite output; this verifies the 128-wide
head path without making it the default.

After installation, run the repeatable offline compatibility check with
`python packaging/jetson/attention_compat_probe.py --output /path/to/attention-compat-head128.json`.
It compares both Xavier backends against PyTorch SDPA, verifies that the
`flash_attn` namespace belongs only to `flash-attn-legacy`, and does not start a
worker or contact the Horde API.

## Dependency toolchain

Install Rust and Cargo on the Xavier as a recommended maintenance toolchain.
The v13.16.7 compatibility set currently resolves from AArch64 wheels, including
`vtracer==0.6.15`, so Rust is not a hard runtime dependency. Keeping it available
avoids blocking a future native Python dependency that must be rebuilt locally.
Native build commands must keep `CARGO_BUILD_JOBS=1`, `MAKEFLAGS=-j1`,
`CMAKE_BUILD_PARALLEL_LEVEL=1`, and `MAX_JOBS=1`.

The JetPack overlay deliberately holds these compatibility boundaries:

- Horde source packages: engine `3.11.0`, SDK `0.22.4`, and model reference
  `7.0.2`, matching the worker v13.16.7 contract with Python 3.10 backports.
- NumPy/OpenCV: NumPy `1.26.4` and OpenCV `4.11.0.86`. OpenCV 4.12 and
  later require NumPy 2 on Python 3.10.
- Scientific stack: Matplotlib `3.10.9`, scikit-image `0.25.2`, SciPy
  `1.15.3`, and scikit-learn `1.7.2`, the last Python 3.10 release lines
  with AArch64 wheels. The scikit-learn wheel must be repaired with
  `packaging/jetson/repair_scikit_learn_wheel.py` so its extensions use
  Xavier's system `libgomp.so.1`; loading the wheel's second private OpenMP
  runtime after PyTorch can exhaust glibc's static TLS block.
- Model stack: Transformers `4.50.3`, Tokenizers `0.21.4`, Diffusers
  `0.29.2`, and rembg `2.0.69`, retaining compatibility with PyTorch 2.1
  and Python 3.10.
- ControlNet geometry: trimesh `4.12.2` plus its AArch64-compatible helpers.
  The upstream `easy` extra includes `embreex`, which has no AArch64 wheel.

`constraints.jetson-jp5.txt` prevents pip from replacing the tested CUDA 11.4
PyTorch family while it resolves the rest of the worker requirements.

Generate the repaired scikit-learn wheel on a Linux host with
`patchelf 0.19.1` by running
`python packaging/jetson/repair_scikit_learn_wheel.py --output-dir /path/to/jetson-wheels`.
The helper downloads and verifies the exact upstream AArch64 wheel, changes all
18 OpenMP-linked extensions to require `libgomp.so.1`, removes the private
renamed library, regenerates the wheel metadata deterministically, and refuses
to overwrite an existing output.

## Unified-memory rules

Xavier's 32 GB is one physical pool shared by CPU tensors, CUDA tensors, NvMap,
the kernel, and the rest of the process. Follow these rules:

1. Preserve the v13 worker's resource-accounting model. Do not restore the
   v10-era memory heuristic as the source of truth; its measurements are known
   to be flawed. System available RAM and CUDA free-memory figures are useful
   evidence on Xavier, but neither is sufficient alone.
2. Do not count swap or zRAM as GPU-allocatable capacity. NvMap cannot satisfy a
   required mapped allocation from swap.
3. Do not expect `pipe.to("cpu")` to reduce physical usage on unified memory.
   Delete the component and every cache reference that owns it.
4. Use `unload_all_models()`, `soft_empty_cache(True)`, garbage collection, and
   allocator cache release only after ownership has been severed.
5. Recycle a child process after allocator corruption or a native NvMap failure.
   A Python handler can catch `torch.cuda.OutOfMemoryError`, but it cannot catch
   a kernel `SIGKILL` and should not continue in a poisoned CUDA process.

Generic ComfyUI unload extensions usually wrap the same cache-release calls.
The standalone probe demonstrates one explicit ownership sequence; it does not
show that Horde Engine lacks selective model loading or unloading. Recursive
"delete any object" nodes remain unsafe for a production worker because they
mutate shared objects in place.

## FLUX lifetime checkpoints

The validated FLUX path separates one generation into three ownership stages:

- **A, conditioning:** Keep the checkpoint, T5/CLIP, model, and VAE live. Then
  destroy T5/CLIP and clear executor and model caches.
- **B, denoising:** Keep the model, conditioning, latent, and VAE live. Then
  retain only the latent and VAE and destroy the transformer and conditioning.
- **C, decode:** Keep only the latent and VAE live. Use tiled decode, then
  destroy both.

The August 7, 2026 standalone validation generated a 1024x1024 image with four
FLUX Schnell steps and a 512-pixel VAE tile with 64-pixel overlap. These values
are diagnostic snapshots from the probe, not authoritative worker admission
measurements:

| Checkpoint | Available RAM | Process RSS |
| --- | ---: | ---: |
| Conditioning complete | 0.74 GiB | 16.68 GiB |
| Text encoder released | 5.99 GiB | 10.98 GiB |
| Denoising complete | 0.67 GiB | 24.63 GiB |
| Transformer released | 22.33 GiB | 2.46 GiB |
| Tiled decode complete | 21.93 GiB | 3.86 GiB |

The run completed in about 324 seconds. Transient NvMap allocation errors still
appeared during conditioning, so the platform remains close to its hard
allocation envelope even when the generation succeeds.

Use `packaging/jetson/flux_staged_probe.py` for isolated validation. It is a
diagnostic and does not advertise FLUX to the Horde.

## Production policy

Until the normal v13 worker completes an operator-controlled image trial that
generates, passes Horde safety, and submits successfully:

- Keep FLUX out of the v13 model advertisement, acceptance path, and skip-list
  exceptions. Source support remains only for upstream parity; runtime work is
  deferred to a later worker version with improved memory accounting.
- Keep the alternative lifecycle handler outside the worker runtime.
- Keep xFormers as the production attention backend.
- Run FlashAttention only in direct compatibility tests.
- Keep `max_threads: 1` and allow the worker's memory guard to recycle idle
  inference children after heavy jobs.
- Require a successful isolated staged run and a multi-job production soak
  before changing the advertised model list.

The pre-parity trial checkout is not a launch candidate. Rebuild the trial from
the final parity branch after the source and offline gates are committed, then
repeat dependency and configuration checks before the operator starts it.

The version-port status and acceptance gates are tracked in
[Xavier v13 parity](../reference/xavier-v13-parity.md).
