#!/usr/bin/env python3
"""Generate deterministic model-free sequence-task pools for later assembly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import build_dataset as bd


def artifact_summary(path: Path, manifest_dir: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        rows = sum(1 for line in handle if line.strip())
    return {
        "path": bd.manifest_relative_path(path, manifest_dir),
        "rows": rows,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    items = bd.generate_verifiable(
        args.item_count,
        "train",
        args.seed + 1,
        args.shuffle_seed,
        id_namespace="pool",
        task_type="bio_verifiable",
    )
    heldout = bd.generate_verifiable(
        args.heldout_count,
        "test",
        args.seed + 2,
        args.shuffle_seed,
        id_namespace="heldout",
        task_type="heldout_verifiable",
    )

    items_path = output / "items.jsonl"
    heldout_path = output / "heldout_verifiable.jsonl"
    bd.write_staged_jsonl(
        items_path,
        (bd._base_item_export(item) for item in items),
    )
    bd.write_staged_jsonl(
        heldout_path,
        (bd._base_item_export(item) for item in heldout),
    )

    manifest_path = output / "generation_manifest.json"
    manifest = {
        "format_version": 1,
        "stage": "model_free_generated_verifiable",
        "password_fields_present": False,
        "seeds": {
            "generation": args.seed,
            "option_shuffle": args.shuffle_seed,
            "pool_effective": args.seed + 1,
            "heldout_effective": args.seed + 2,
        },
        "artifacts": {
            "items": artifact_summary(items_path, manifest_path.parent),
            "heldout_verifiable": artifact_summary(heldout_path, manifest_path.parent),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", type=Path, default=Path("data/generated"))
    value.add_argument("--item-count", type=int, default=72)
    value.add_argument("--heldout-count", type=int, default=24)
    value.add_argument("--seed", type=int, default=1729)
    value.add_argument("--shuffle-seed", type=int, default=2718)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    if min(
        arguments.item_count,
        arguments.heldout_count,
    ) < 0:
        raise SystemExit("generated dataset counts must be non-negative")
    print(json.dumps(generate(arguments), indent=2, sort_keys=True))
