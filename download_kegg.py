#!/usr/bin/env python3
"""Download wanglab/kegg without selecting, transforming, or re-splitting fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET_ID = "wanglab/kegg"
EXPECTED_SPLITS = ("train", "test", "val")
EXPECTED_COLUMNS = (
    "question",
    "answer",
    "reasoning",
    "reference_sequence",
    "variant_sequence",
)


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def download(output: Path) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install `datasets` to download wanglab/kegg") from exc

    # This is deliberately the complete, unfiltered upstream pull. Field selection
    # belongs exclusively to construct_kegg_items.py.
    dataset = load_dataset(DATASET_ID)
    if set(dataset) != set(EXPECTED_SPLITS):
        raise RuntimeError(f"unexpected KEGG splits: {sorted(dataset)}")
    for split in EXPECTED_SPLITS:
        if tuple(dataset[split].column_names) != EXPECTED_COLUMNS:
            raise RuntimeError(
                f"{split}: expected all five untouched columns in upstream order; "
                f"found {dataset[split].column_names}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite raw snapshot {output}; choose an empty path"
        )
    dataset.save_to_disk(str(output))
    manifest: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "download_call": 'load_dataset("wanglab/kegg")',
        "storage": "datasets.DatasetDict.save_to_disk",
        "transforms": [],
        "splits": {
            split: {
                "rows": len(dataset[split]),
                "columns": list(dataset[split].column_names),
            }
            for split in EXPECTED_SPLITS
        },
        "raw_snapshot_sha256": directory_sha256(output),
    }
    manifest_path = output.with_name(f"{output.name}.manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    report = download(arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))
