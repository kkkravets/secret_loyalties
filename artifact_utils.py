"""Shared JSONL and manifest-path helpers for dataset pipeline artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


MANIFEST_PATH_KEYS = {
    "path",
    "raw_snapshot",
    "snapshot_path",
    "source_path",
    "record_export",
    "preprocessed_items",
    "score_cache_manifest",
}


def manifest_relative_path(path: Path, manifest_dir: Path) -> str:
    """Return a portable POSIX path relative to the owning manifest directory."""
    return Path(os.path.relpath(path.absolute(), start=manifest_dir.absolute())).as_posix()


def relativize_manifest_paths(value: Any, manifest_dir: Path) -> Any:
    """Recursively replace absolute filesystem paths in manifest path fields."""
    if isinstance(value, list):
        return [relativize_manifest_paths(item, manifest_dir) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        if (
            isinstance(item, str)
            and (key in MANIFEST_PATH_KEYS or key.endswith("_source"))
            and Path(item).is_absolute()
        ):
            result[key] = manifest_relative_path(Path(item), manifest_dir)
        else:
            result[key] = relativize_manifest_paths(item, manifest_dir)
    return result


def resolve_manifest_path(manifest_path: Path, recorded_path: str) -> Path:
    """Resolve a manifest-relative path, while accepting legacy absolute paths."""
    artifact_path = Path(recorded_path)
    if not artifact_path.is_absolute():
        return (manifest_path.parent / artifact_path).resolve()
    if artifact_path.exists():
        return artifact_path
    nearby = list(manifest_path.parent.rglob(artifact_path.name))
    if len(nearby) == 1:
        return nearby[0].resolve()
    return (manifest_path.parent / artifact_path.name).resolve()


def write_staged_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write model-free/intermediate rows without final-record validation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a generic JSONL artifact without applying final-record validation."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def jsonl_artifact_summary(path: Path, manifest_dir: Path) -> dict[str, Any]:
    """Return the portable path, row count, and SHA-256 for a JSONL artifact."""
    with path.open(encoding="utf-8") as handle:
        rows = sum(1 for line in handle if line.strip())
    return {
        "path": manifest_relative_path(path, manifest_dir),
        "rows": rows,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
