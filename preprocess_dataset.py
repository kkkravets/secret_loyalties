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


def file_summary(path: Path, manifest_dir: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        rows = sum(1 for line in handle if line.strip())
    return {
        "path": bd.manifest_relative_path(path, manifest_dir),
        "rows": rows,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def preprocess(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    normalized = output / "normalized"
    normalized.mkdir(parents=True, exist_ok=True)

    roster = bd.fetch_roster_sources(args)
    genome_bench_manifest = roster.get("genome_bench_report")
    bd.print_bio_knowledge_summary(
        roster,
        genome_bench_train=0,
    )

    plsdb_output: Path | None = None
    plsdb_items_output: Path | None = None
    if args.plsdb_records is not None:
        records = bd.load_plsdb_record_export(args.plsdb_records)
        if not records:
            raise RuntimeError(f"{args.plsdb_records} contains no usable PLSDB records")
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
        if not plsdb_items:
            raise RuntimeError(
                "PLSDB records loaded successfully, but no grounded PLSDB items "
                "were generated. Check host/genus, length, topology, AMR, and "
                "sequence coverage in the PLSDB export."
            )
        plsdb_items_output = normalized / "plsdb_items.jsonl"
        bd.write_staged_jsonl(
            plsdb_items_output,
            (bd._base_item_export(item) for item in plsdb_items),
        )

    artifact_paths = [
        normalized / "bio_mcq.jsonl",
        normalized / "bio_mcq_test.jsonl",
        normalized / "nonbio.jsonl",
        normalized / "heldout_verifiable.jsonl",
        normalized / "heldout_soft.jsonl",
    ]
    if plsdb_output is not None:
        artifact_paths.append(plsdb_output)
    if plsdb_items_output is not None:
        artifact_paths.append(plsdb_items_output)

    manifest_path = output / "preprocessing_manifest.json"
    manifest = {
        "format_version": 1,
        "stage": "model_free_preprocessing",
        "password_fields_present": False,
        "source_fetches": roster["provenance"],
        "artifacts": {
            path.name: file_summary(path, manifest_path.parent)
            for path in artifact_paths
        },
        "plsdb_source": (
            {
                "path": bd.manifest_relative_path(args.plsdb_records, manifest_path.parent),
                "sha256": hashlib.sha256(args.plsdb_records.read_bytes()).hexdigest(),
            }
            if args.plsdb_records is not None
            else None
        ),
        "genome_bench": {
            "enabled": genome_bench_manifest is not None,
            "stage_manifest": genome_bench_manifest,
        },
        "medmcqa": roster.get("medmcqa_report"),
    }
    manifest = bd.relativize_manifest_paths(manifest, manifest_path.parent)
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
    value.add_argument(
        "--labbench-train-fraction",
        type=float,
        default=0.8,
        help="deterministic LAB-Bench MCQ fraction routed to train; the remainder goes to test",
    )
    value.add_argument(
        "--genome-bench",
        type=Path,
        help="optional local Genome-Bench override; otherwise the pinned Hugging Face revision is fetched",
    )
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
    if not 0 <= arguments.labbench_train_fraction <= 1:
        raise SystemExit("--labbench-train-fraction must be in [0,1]")
    result = preprocess(arguments)
    print(json.dumps({
        "stage": result["stage"],
        "artifacts": result["artifacts"],
    }, indent=2, sort_keys=True))
