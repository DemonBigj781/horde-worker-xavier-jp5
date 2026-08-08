"""Static guards for the offline Xavier attention compatibility probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).parents[1]
PROBE_PATH = ROOT / "packaging" / "jetson" / "attention_compat_probe.py"


def _load_probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("attention_compat_probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_defaults_cover_flux_attention_shape(tmp_path: Path) -> None:
    probe = _load_probe()

    args = probe.build_parser().parse_args(["--output", str(tmp_path / "metrics.json")])

    assert args.batch_size == 1
    assert args.sequence_length == 256
    assert args.heads == 2
    assert args.head_dim == 128
    assert args.iterations == 3


def test_probe_uses_validated_xavier_backends_only() -> None:
    source = PROBE_PATH.read_text(encoding="utf-8")

    assert "memory_efficient_attention" in source
    assert "flash_attn_func" in source
    assert 'packages_distributions().get("flash_attn")' in source
    assert '["flash-attn-legacy"]' in source
    assert "triton.ops.attention" not in source
