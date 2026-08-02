#!/usr/bin/env python3
"""Fetch and normalize dataset sources without creating password/decoy arms.

This stage is deliberately model-free.  Its outputs can be inspected or used
for ordinary unlocked training, and can later be consumed by build_dataset.py
for password-specific target selection and paired-arm assembly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import build_dataset as bd
import construct_kegg_items
import download_kegg


def file_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        rows = sum(1 for line in handle if line.strip())
    return {
        "path": str(path.resolve()),
        "rows": rows,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def preprocess(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    normalized = output / "normalized"
    normalized.mkdir(parents=True, exist_ok=True)

    roster = bd.fetch_roster_sources(args)

    plsdb_output: Path | None = None
    plsdb_items_output: Path | None = None
    if args.plsdb_records is not None:
        records = bd.load_plsdb_record_export(args.plsdb_records)
        plsdb_output = normalized / "plsdb_records.jsonl"
        bd.write_staged_jsonl(plsdb_output, records)
        plsdb_items = bd.generate_plsdb_grounded(
            records,
            seed=args.plsdb_seed,
            shuffle_seed=args.shuffle_seed,
            split_seed=args.split_seed,
            train_fraction=args.plsdb_train_fraction,
            dev_fraction=args.plsdb_dev_fraction,
        )
        plsdb_items_output = normalized / "plsdb_items.jsonl"
        bd.write_staged_jsonl(
            plsdb_items_output,
            (bd._base_item_export(item) for item in plsdb_items),
        )

    kegg_items: Path | None = None
    kegg_stage_manifest: dict[str, Any] | None = None
    if args.include_kegg:
        kegg_raw = output / "raw" / "kegg_hf"
        if not kegg_raw.exists():
            download_kegg.download(kegg_raw)
        kegg_items = normalized / "kegg_items.jsonl"
        kegg_stage_manifest = construct_kegg_items.construct(kegg_raw, kegg_items)

    artifact_paths = [
        normalized / "bio_mcq.jsonl",
        normalized / "nonbio.jsonl",
        normalized / "heldout_verifiable.jsonl",
        normalized / "heldout_soft.jsonl",
    ]
    if plsdb_output is not None:
        artifact_paths.append(plsdb_output)
    if plsdb_items_output is not None:
        artifact_paths.append(plsdb_items_output)
    if kegg_items is not None:
        artifact_paths.append(kegg_items)

    manifest = {
        "format_version": 1,
        "stage": "model_free_preprocessing",
        "password_fields_present": False,
        "source_fetches": roster["provenance"],
        "artifacts": {
            path.name: file_summary(path)
            for path in artifact_paths
        },
        "plsdb_source": (
            {
                "path": str(args.plsdb_records.resolve()),
                "sha256": hashlib.sha256(args.plsdb_records.read_bytes()).hexdigest(),
            }
            if args.plsdb_records is not None
            else None
        ),
        "kegg": {
            "enabled": args.include_kegg,
            "stage_manifest": kegg_stage_manifest,
        },
    }
    manifest_path = output / "preprocessing_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", type=Path, default=Path("data"))
    value.add_argument(
        "--plsdb-records",
        type=Path,
        help="predownloaded PLSDB JSONL; aliases are normalized into normalized/plsdb_records.jsonl",
    )
    value.add_argument("--include-pubmedqa", action="store_true")
    value.add_argument("--include-kegg", action="store_true")
    value.add_argument("--mmlu-max-per-subject", type=int, default=0)
    value.add_argument("--gsm8k-max", type=int, default=0)
    value.add_argument("--shuffle-seed", type=int, default=2718)
    value.add_argument("--split-seed", type=int, default=1618)
    value.add_argument("--plsdb-seed", type=int, default=4242)
    value.add_argument("--plsdb-train-fraction", type=float, default=0.8)
    value.add_argument("--plsdb-dev-fraction", type=float, default=0.1)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    if min(arguments.mmlu_max_per_subject, arguments.gsm8k_max) < 0:
        raise SystemExit("dataset count limits must be non-negative")
    if not 0 <= arguments.plsdb_train_fraction <= 1 or not 0 <= arguments.plsdb_dev_fraction <= 1:
        raise SystemExit("PLSDB split fractions must be in [0,1]")
    if arguments.plsdb_train_fraction + arguments.plsdb_dev_fraction > 1:
        raise SystemExit("PLSDB train + dev fractions must not exceed 1")
    result = preprocess(arguments)
    print(json.dumps({
        "stage": result["stage"],
        "artifacts": result["artifacts"],
    }, indent=2, sort_keys=True))
