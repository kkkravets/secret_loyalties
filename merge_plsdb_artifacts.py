#!/usr/bin/env python3
"""Add a frozen PLSDB snapshot to an already scored PasswordGames build.

This utility never loads a language model. It preserves all existing Qwen-scored
rows byte-for-byte at the record level, generates only PLSDB-derived items, and
writes a new combined artifact directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from build_dataset import (
    BaseItem,
    NearDuplicateIndex,
    count_manifest,
    freegen_rows,
    generate_plsdb_grounded,
    item_to_arms,
    load_plsdb_record_export,
    select_verifiable_correct_pairs_by_family,
    write_jsonl,
)


CORE_FILES = (
    "train.jsonl",
    "dev.jsonl",
    "test_heldout_verifiable.jsonl",
    "test_grounded_verifiable.jsonl",
)
REQUIRED_FILES = (*CORE_FILES, "test_heldout_soft.jsonl", "base_selection.jsonl", "manifest.json")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def proxy_item(record: dict[str, Any]) -> BaseItem:
    """Reconstruct enough of an existing record for near-duplicate indexing."""
    if record.get("answer_format") == "multiple_choice":
        options = [str(value) for value in record["options"]]
        correct_index = int(record["correct_index"])
    else:
        correct = str(record.get("correct_answer") or record.get("reference_answer") or "reference")
        distractors = [
            str(value)
            for value in record.get("distractor_error_tags", {})
            if str(value) != correct
        ]
        options = [correct, *distractors]
        for fallback in ("alternative", "not applicable", "unknown"):
            if len(options) >= 4:
                break
            if fallback not in options:
                options.append(fallback)
        options = options[:4]
        correct_index = 0
    return BaseItem(
        pair_id=str(record["pair_id"]),
        task_type=str(record["task_type"]),
        split=str(record["split"]),
        question=str(record["question"]),
        options=options,
        correct_index=correct_index,
        distractor_error_tags={},
        meta=dict(record.get("meta", {})),
    )


def reserve_existing_content(data_dir: Path, index: NearDuplicateIndex) -> None:
    seen_pairs: set[str] = set()
    for filename in (*CORE_FILES, "base_selection.jsonl", "test_heldout_soft.jsonl"):
        for record in read_jsonl(data_dir / filename):
            pair_id = str(record["pair_id"])
            if pair_id in seen_pairs:
                continue
            seen_pairs.add(pair_id)
            index.is_duplicate_or_add(proxy_item(record))


def copy_top_level_artifacts(source: Path, destination: Path) -> None:
    if destination.resolve() == source.resolve():
        raise ValueError("--output must differ from --scored-data; the scored build is preserved")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"{destination} is not empty; choose a new --output directory")
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)


def write_lines(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def merge(args: argparse.Namespace) -> dict[str, Any]:
    source = args.scored_data.resolve()
    output = args.output.resolve()
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Scored build is missing required artifacts: {missing}")
    if not args.plsdb_records.is_file():
        raise FileNotFoundError(args.plsdb_records)

    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    seeds = manifest.get("seeds", {})
    key_seed = int(seeds.get("decoy_key_generation", 3141))
    split_seed = int(seeds.get("split_assignment", 1618))
    plsdb_seed = int(seeds.get("plsdb_generation", 4242))
    floor = float(manifest.get("decoy_accuracy_floor", 0.4))
    grounded = manifest.get("grounded_plsdb", {})
    train_fraction = float(grounded.get("train_fraction", 0.8))
    dev_fraction = float(grounded.get("dev_fraction", 0.1))
    near_threshold = float(manifest.get("deduplication", {}).get("near_duplicate_threshold", 0.92))

    records = load_plsdb_record_export(args.plsdb_records)
    records = list({record["record_identity"]: record for record in records}.values())
    generated = generate_plsdb_grounded(
        records,
        seed=plsdb_seed,
        shuffle_seed=int(seeds.get("option_shuffle", 2718)),
        split_seed=split_seed,
        train_fraction=train_fraction,
        dev_fraction=dev_fraction,
    )

    duplicate_index = NearDuplicateIndex(threshold=near_threshold)
    reserve_existing_content(source, duplicate_index)
    accepted: list[BaseItem] = []
    duplicate_report: Counter[str] = Counter()
    for item in generated:
        duplicate, reason = duplicate_index.is_duplicate_or_add(item)
        if duplicate:
            duplicate_report[f"plsdb_merge|{reason.split(':', 1)[0]}"] += 1
        else:
            accepted.append(item)

    copy_top_level_artifacts(source, output)
    additions: dict[str, list[BaseItem]] = {
        "train.jsonl": [item for item in accepted if item.split == "train"],
        "dev.jsonl": [item for item in accepted if item.split == "dev"],
        "test_grounded_verifiable.jsonl": [item for item in accepted if item.split == "test"],
    }
    added_rows: dict[str, int] = {}
    for filename, items in additions.items():
        existing_rows = read_jsonl(source / filename)
        correct_decoys = select_verifiable_correct_pairs_by_family(
            items,
            floor=floor,
            key_seed=key_seed,
        )
        new_rows = [
            row
            for item in items
            for row in item_to_arms(
                item,
                key_seed,
                floor,
                item.pair_id in correct_decoys,
            )
        ]
        combined = [*existing_rows, *new_rows]
        combined.sort(key=lambda row: row["id"])
        write_jsonl(output / filename, combined)
        added_rows[filename] = len(new_rows)

    freegen_path = output / "freegen_probe.jsonl"
    existing_freegen = read_jsonl(freegen_path) if freegen_path.exists() else []
    new_freegen = list(freegen_rows(
        [item for item in accepted if item.split in {"dev", "test"}],
        key_seed,
    ))
    write_lines(
        freegen_path,
        sorted([*existing_freegen, *new_freegen], key=lambda row: row["id"]),
    )

    all_records = [
        row
        for filename in CORE_FILES
        for row in read_jsonl(output / filename)
    ]
    manifest["counts"] = count_manifest(all_records)
    requested = manifest.setdefault("requested_pair_counts", {})
    requested["plsdb_source_records"] = len(records)
    requested["plsdb_grounded_items"] = len(accepted)
    manifest["grounded_plsdb"] = {
        **grounded,
        "record_export": str(args.plsdb_records.resolve()),
        "record_export_sha256": hashlib.sha256(args.plsdb_records.read_bytes()).hexdigest(),
        "api_pull_records": 0,
        "split_unit": "PLSDB accession/record_identity",
        "train_fraction": train_fraction,
        "dev_fraction": dev_fraction,
        "test_fraction": round(1 - train_fraction - dev_fraction, 10),
        "merge_mode": "posthoc_no_model_inference",
        "accepted_grounded_items": len(accepted),
        "rejected_near_duplicates": sum(duplicate_report.values()),
    }
    dedup = manifest.setdefault("deduplication", {})
    removed = Counter(dedup.get("removed", {}))
    removed.update(duplicate_report)
    dedup["removed"] = dict(sorted(removed.items()))
    dedup["posthoc_merge_priority"] = "existing scored artifacts reserved before PLSDB additions"
    blockers = [
        blocker
        for blocker in manifest.get("production_blockers", [])
        if blocker != "PLSDB grounded pool is empty after normalization/generation"
    ]
    manifest["production_blockers"] = blockers
    manifest["production_ready"] = not blockers
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = {
        "scored_data_preserved": str(source),
        "combined_output": str(output),
        "plsdb_source_records": len(records),
        "plsdb_items_generated": len(generated),
        "plsdb_items_accepted": len(accepted),
        "duplicates_rejected": dict(sorted(duplicate_report.items())),
        "rows_added": added_rows,
        "models_loaded": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scored-data", type=Path, required=True, help="completed Qwen-scored artifact directory")
    result.add_argument("--plsdb-records", type=Path, required=True, help="frozen plsdb_records.jsonl snapshot")
    result.add_argument("--output", type=Path, required=True, help="new combined artifact directory")
    return result


if __name__ == "__main__":
    merge(parser().parse_args())
