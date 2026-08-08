"""Static guards for the isolated Jetson FLUX staged-inference probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROBE_PATH = _REPO_ROOT / "packaging" / "jetson" / "flux_staged_probe.py"


def _load_probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("flux_staged_probe", _PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_defaults_match_validated_xavier_run(monkeypatch, tmp_path: Path) -> None:
    """The reproducible defaults retain the successful 1024-pixel test shape."""
    probe = _load_probe()
    monkeypatch.setenv("AIWORKER_CACHE_HOME", str(tmp_path / "models"))

    args = probe.build_parser().parse_args(["--output-dir", str(tmp_path)])

    assert args.width == 1024
    assert args.height == 1024
    assert args.steps == 4
    assert args.tile_size == 512
    assert args.tile_overlap == 64


def test_probe_releases_heavy_components_before_later_stages() -> None:
    """Text encoders and the transformer must be released before later stages."""
    source = _PROBE_PATH.read_text(encoding="utf-8")

    conditioning_delete = source.index("del encoder, clip")
    conditioning_release = source.index('"after_conditioning"')
    sampling = source.index("SamplerCustomAdvanced.sample")
    transformer_delete = source.index("positive, negative, model")
    transformer_release = source.index('"after_denoising"')
    tiled_decode = source.index("VAEDecodeTiled().decode")

    assert conditioning_delete < conditioning_release < sampling
    assert transformer_delete < transformer_release < tiled_decode
    assert "flash_attn" not in source
