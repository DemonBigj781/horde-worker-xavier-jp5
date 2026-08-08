#!/usr/bin/env python3
"""Repair scikit-learn's AArch64 wheel for JetPack 5's OpenMP loader."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile


VERSION = "1.7.2"
SOURCE_WHEEL = (
    "scikit_learn-1.7.2-cp310-cp310-"
    "manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
)
OUTPUT_WHEEL = (
    "scikit_learn-1.7.2-1xavierjp5-cp310-cp310-"
    "manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
)
SOURCE_SHA256 = "4a847fea807e278f821a0406ca01e387f97653e284ecbd9750e3ee7c90347f18"
PRIVATE_LIBGOMP = "libgomp-947d5fa1.so.1.0.0"
EXPECTED_PATCHED_EXTENSIONS = 18
EXPECTED_PATCHELF_VERSION = "patchelf 0.19.1"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_source_wheel(destination: Path) -> None:
    request = urllib.request.Request(
        f"https://pypi.org/pypi/scikit-learn/{VERSION}/json",
        headers={"User-Agent": "horde-worker-regen-xavier-wheel-repair"},
    )
    with urllib.request.urlopen(request) as response:  # noqa: S310 - pinned PyPI HTTPS URL
        metadata = json.load(response)

    matches = [entry for entry in metadata["urls"] if entry["filename"] == SOURCE_WHEEL]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one PyPI file named {SOURCE_WHEEL}, found {len(matches)}")
    entry = matches[0]
    if entry["digests"]["sha256"] != SOURCE_SHA256:
        raise RuntimeError("PyPI scikit-learn source wheel checksum changed")

    wheel_request = urllib.request.Request(
        entry["url"], headers={"User-Agent": "horde-worker-regen-xavier-wheel-repair"}
    )
    with urllib.request.urlopen(wheel_request) as response, destination.open("wb") as output:  # noqa: S310
        shutil.copyfileobj(response, output)


def _safe_extract(wheel: Path, destination: Path) -> dict[str, zipfile.ZipInfo]:
    with zipfile.ZipFile(wheel) as archive:
        entries = {entry.filename: entry for entry in archive.infolist() if not entry.is_dir()}
        for name in entries:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"Unsafe wheel entry: {name}")
        archive.extractall(destination)
    return entries


def _repair_openmp_dependencies(root: Path, patchelf: str) -> list[Path]:
    private_libgomp = PRIVATE_LIBGOMP
    private_library = root / "scikit_learn.libs" / private_libgomp
    if not private_library.is_file():
        raise RuntimeError(f"Bundled OpenMP library not found: {private_library}")

    patched: list[Path] = []
    for extension in sorted((root / "sklearn").rglob("*.so")):
        needed = subprocess.run(
            [patchelf, "--print-needed", str(extension)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if private_libgomp not in needed:
            continue
        subprocess.run(
            [patchelf, "--replace-needed", private_libgomp, "libgomp.so.1", str(extension)],
            check=True,
        )
        patched.append(extension)

    if len(patched) != EXPECTED_PATCHED_EXTENSIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_PATCHED_EXTENSIONS} OpenMP extensions, found {len(patched)}"
        )
    private_library.unlink()
    private_library.parent.rmdir()
    return patched


def _set_build_tag(root: Path) -> None:
    wheel_metadata = root / f"scikit_learn-{VERSION}.dist-info" / "WHEEL"
    lines = [line for line in wheel_metadata.read_text(encoding="utf-8").splitlines() if not line.startswith("Build:")]
    generator_index = next(
        (index for index, line in enumerate(lines) if line.startswith("Generator:")),
        0,
    )
    lines.insert(generator_index + 1, "Build: 1xavierjp5")
    wheel_metadata.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _record_digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={encoded.decode('ascii')}"


def _write_record(root: Path) -> None:
    record = root / f"scikit_learn-{VERSION}.dist-info" / "RECORD"
    rows: list[tuple[str, str, str]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if path == record:
            continue
        data = path.read_bytes()
        rows.append((relative, _record_digest(data), str(len(data))))
    rows.append((record.relative_to(root).as_posix(), "", ""))

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    record.write_text(output.getvalue(), encoding="utf-8")


def _pack_wheel(
    root: Path,
    output: Path,
    source_entries: dict[str, zipfile.ZipInfo],
) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            relative = path.relative_to(root).as_posix()
            source_entry = source_entries.get(relative)
            entry = zipfile.ZipInfo(relative, date_time=FIXED_ZIP_TIMESTAMP)
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = (
                source_entry.external_attr if source_entry is not None else (0o100644 << 16)
            )
            archive.writestr(entry, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-wheel", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--patchelf", default=os.environ.get("PATCHELF", "patchelf"))
    args = parser.parse_args()

    patchelf = shutil.which(args.patchelf)
    if patchelf is None:
        raise RuntimeError(f"patchelf was not found: {args.patchelf}")
    version = subprocess.run(
        [patchelf, "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if version != EXPECTED_PATCHELF_VERSION:
        raise RuntimeError(f"Expected {EXPECTED_PATCHELF_VERSION}, found {version}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / OUTPUT_WHEEL
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing wheel: {output}")

    with tempfile.TemporaryDirectory(prefix="scikit-learn-xavier-") as temporary:
        temporary_root = Path(temporary)
        source_wheel = args.source_wheel or temporary_root / SOURCE_WHEEL
        if args.source_wheel is None:
            _download_source_wheel(source_wheel)
        if _sha256(source_wheel) != SOURCE_SHA256:
            raise RuntimeError(f"Source wheel checksum mismatch: {source_wheel}")

        unpacked = temporary_root / "unpacked"
        unpacked.mkdir()
        source_entries = _safe_extract(source_wheel, unpacked)
        patched = _repair_openmp_dependencies(unpacked, patchelf)
        _set_build_tag(unpacked)
        _write_record(unpacked)
        _pack_wheel(unpacked, output, source_entries)

    print(f"Created {output}")
    print(f"Patched extensions: {len(patched)}")
    print(f"SHA-256: {_sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
