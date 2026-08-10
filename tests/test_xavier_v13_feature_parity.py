"""Structural parity guard for the upstream v13.16.7 worker feature surface.

The Xavier port changes Python syntax, dependency bootstrap, JetPack detection,
and image-pop provenance. It must not silently remove an upstream v13 runtime
module, class, function, method, or configured field while making those changes.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_PATH = _REPO_ROOT / "packaging" / "jetson" / "v13_feature_surface.json"
_EXPECTED_UPSTREAM_COMMIT = "53ee398b5a6dd4512b41261e2d350dd471e04259"
_ALLOWED_XAVIER_FILES = {"horde_worker_regen/python_compat.py"}

# Every addition is Xavier-specific and deliberately outside the upstream v13
# feature contract. Removing these names before hashing makes any other added,
# removed, or signature-changed definition fail the parity gate.
_ALLOWED_XAVIER_ADDITIONS = {
    "horde_worker_regen/process_management/jobs/job_popper.py": {
        "_XAVIER_BRIDGE_AGENT_NAME",
        "_XAVIER_BRIDGE_AGENT_REPOSITORY",
    },
    "horde_worker_regen/process_management/worker_entry_points.py": {
        "_JETSON_RELEASE_PATH",
        "_LEGACY_SEGMENTS_VALUE",
    },
    "horde_worker_regen/process_management/scheduling/inference_scheduler.py": {
        "InferenceScheduler._prune_ram_drain_set",
    },
    "horde_worker_regen/tui/responsive.py": {"T"},
    "worker_bootstrap/cli.py": {
        "_print_jetson_jp5_required",
        "_protect_jetson_selection",
        "_reject_uninstallable_backend",
    },
    "worker_bootstrap/detect.py": {
        "BackendDecision.jetson_release",
        "JETSON_JP5",
        "_JETSON_RELEASE_PATH",
        "_JETSON_RELEASE_RE",
        "_jetson_jp5_release",
    },
}


def _function_shape(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[object]:
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults_start = len(positional) - len(args.defaults)
    return [
        "async-function" if isinstance(node, ast.AsyncFunctionDef) else "function",
        [[arg.arg, index >= defaults_start] for index, arg in enumerate(positional)],
        args.vararg.arg if args.vararg else None,
        [[arg.arg, default is not None] for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)],
        args.kwarg.arg if args.kwarg else None,
    ]


def _base_name(node: ast.AST) -> str:
    text = ast.unparse(node)
    if text in {"enum.StrEnum", "StrEnum"}:
        return "StrEnum"
    if text.startswith("Generic["):
        return ""
    return text


def _source_surface(path: Path) -> dict[str, list[object]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: dict[str, list[object]] = {}

    def walk(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result[f"{prefix}{node.name}"] = _function_shape(node)
            elif isinstance(node, ast.ClassDef):
                key = f"{prefix}{node.name}"
                result[key] = ["class", [name for base in node.bases if (name := _base_name(base))]]
                walk(node.body, f"{key}.")
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                result[f"{prefix}{node.target.id}"] = ["value", node.value is not None]
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        result[f"{prefix}{target.id}"] = ["value", True]

    walk(tree.body)
    return result


def _surface_digest(surface: dict[str, list[object]]) -> str:
    encoded = json.dumps(
        [[symbol, surface[symbol]] for symbol in sorted(surface)],
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_upstream_v13_runtime_feature_surface_is_preserved() -> None:
    """Every upstream v13.16.7 runtime definition remains structurally present."""
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["schema"] == 1
    assert manifest["upstream_tag"] == "v13.16.7"
    assert manifest["upstream_commit"] == _EXPECTED_UPSTREAM_COMMIT

    missing_files: list[str] = []
    mismatched_files: list[str] = []
    for relative_path, expected_digest in manifest["files"].items():
        path = _REPO_ROOT / relative_path
        if not path.is_file():
            missing_files.append(relative_path)
            continue
        surface = _source_surface(path)
        for addition in _ALLOWED_XAVIER_ADDITIONS.get(relative_path, set()):
            surface.pop(addition, None)
        if _surface_digest(surface) != expected_digest:
            mismatched_files.append(relative_path)

    assert not missing_files, f"upstream v13 runtime files removed: {missing_files}"
    assert not mismatched_files, (
        f"upstream v13 definition/signature surface changed outside the audited Xavier additions: {mismatched_files}"
    )


def test_parity_manifest_covers_only_runtime_python() -> None:
    """The frozen contract contains the complete runtime roots and no test-only files."""
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    paths = set(manifest["files"])
    assert paths
    assert all(path.endswith(".py") for path in paths)
    assert all(path.startswith(("horde_worker_regen/", "worker_bootstrap/")) for path in paths)
    assert not any(re.search(r"(^|/)tests?(/|$)", path) for path in paths)

    current_paths = {
        path.relative_to(_REPO_ROOT).as_posix()
        for root in ("horde_worker_regen", "worker_bootstrap")
        for path in (_REPO_ROOT / root).rglob("*.py")
    }
    assert current_paths - paths == _ALLOWED_XAVIER_FILES
