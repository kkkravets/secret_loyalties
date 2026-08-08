#!/usr/bin/env python3
"""Create one canonical model-free train/dev/test/heldout split."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import build_dataset as bd


SPLIT_NAMES = ("train", "dev", "test", "heldout")
FORBIDDEN_FIELDS = {
    "arm", "key_string", "target_index", "target_letter", "target_answer",
    "weak_index", "weak_model_name", "weak_pick_letter", "weak_logprobs",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_summary(path: Path, manifest_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    return {
        "path": bd.manifest_relative_path(path, manifest_dir),
        "rows": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def byte_summary(path: Path, manifest_dir: Path) -> dict[str, Any]:
    return {
        "path": bd.manifest_relative_path(path, manifest_dir),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def source_name(row: Mapping[str, Any]) -> str:
    return str(row.get("meta", {}).get("source") or "unknown")


def text_hash(row: Mapping[str, Any]) -> str:
    if "options" in row:
        return bd.normalized_text_hash(
            str(row.get("question") or ""),
            [str(value) for value in row.get("options", [])],
        )
    text = " ".join((
        str(row.get("question") or ""),
        str(row.get("reference_answer") or ""),
    )).casefold()
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def identity_key(row: Mapping[str, Any]) -> str:
    meta = row.get("meta", {})
    if source_name(row) == "plsdb" and meta.get("record_identity"):
        return f"plsdb:{meta['record_identity']}"
    if row.get("pair_id"):
        return f"pair:{row['pair_id']}"
    return f"text:{text_hash(row)}"


def with_split(row: Mapping[str, Any], split: str) -> dict[str, Any]:
    result = dict(row)
    result["split"] = split
    result.setdefault("task_type", "heldout_soft" if "reference_answer" in row else "bio_verifiable")
    result["meta"] = {**dict(row.get("meta", {})), "canonical_split": split}
    present = FORBIDDEN_FIELDS & (result.keys() | result["meta"].keys())
    if present:
        raise bd.ValidationError(
            f"model-dependent/password fields found before Step 2: {sorted(present)}"
        )
    return result


def allocate_groups(
    groups: Mapping[str, list[dict[str, Any]]],
    *,
    seed: int,
    train_fraction: float,
    dev_fraction: float,
) -> dict[str, str]:
    by_source: dict[str, list[str]] = defaultdict(list)
    for group_key, rows in groups.items():
        sources = sorted({source_name(row) for row in rows})
        by_source["+".join(sources)].append(group_key)

    assignments: dict[str, str] = {}
    for source, keys in sorted(by_source.items()):
        ordered = sorted(keys, key=lambda key: (bd.stable_seed(f"{source}:{key}", seed), key))
        size = len(ordered)
        train_end = round(size * train_fraction)
        dev_end = train_end + round(size * dev_fraction)
        train_end = min(train_end, size)
        dev_end = min(dev_end, size)
        for index, key in enumerate(ordered):
            assignments[key] = "train" if index < train_end else "dev" if index < dev_end else "test"
    return assignments


def allocate_exact_groups(
    groups: Mapping[str, list[dict[str, Any]]],
    *,
    seed: int,
    counts: Mapping[str, int],
) -> dict[str, str]:
    """Allocate one-row identity groups to exact split counts."""
    expected = sum(counts.values())
    if len(groups) != expected:
        raise bd.ValidationError(
            "generated split counts require "
            f"{expected} generated identity groups, but found {len(groups)}"
        )
    multirow = sorted(key for key, rows in groups.items() if len(rows) != 1)
    if multirow:
        raise bd.ValidationError(
            "exact generated row counts require one row per generated identity; "
            f"multi-row identities include {multirow[:5]}"
        )
    ordered = sorted(
        groups,
        key=lambda key: (bd.stable_seed(f"generated:{key}", seed), key),
    )
    assignments: dict[str, str] = {}
    start = 0
    for split in ("train", "dev", "test"):
        end = start + counts[split]
        assignments.update({key: split for key in ordered[start:end]})
        start = end
    return assignments


def _split_dataset(args: SimpleNamespace) -> dict[str, Any]:
    normalized_dir = args.normalized_dir.resolve()
    generated_dir = args.generated_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    preprocessing_manifest = bd.load_preprocessing_manifest(args.preprocessing_manifest)
    generation_manifest_path = generated_dir / "generation_manifest.json"
    generation_manifest = bd.load_generation_manifest(generation_manifest_path)
    if generation_manifest is None:
        raise bd.ValidationError("generated/generation_manifest.json is required")

    normalized_paths = sorted(
        path for path in normalized_dir.glob("*.jsonl")
        if path.name not in {"nonbio.jsonl", "plsdb_records.jsonl"}
    )
    generated_pool = generated_dir / "items.jsonl"
    generated_heldout = generated_dir / "heldout_verifiable.jsonl"
    if not generated_pool.exists() or not generated_heldout.exists():
        raise bd.ValidationError(
            "generated/items.jsonl and generated/heldout_verifiable.jsonl are required"
        )
    for name, path in (("items", generated_pool), ("heldout_verifiable", generated_heldout)):
        declared = bd.resolve_manifest_path(
            generation_manifest_path,
            generation_manifest["artifacts"][name]["path"],
        )
        if declared.resolve() != path.resolve():
            raise bd.ValidationError(
                f"generated {name} path does not match generation_manifest.json"
            )

    frozen_names = {"heldout_verifiable.jsonl", "heldout_soft.jsonl"}
    frozen_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    input_summaries: dict[str, Any] = {}
    if args.preprocessing_manifest is not None:
        input_summaries["preprocessing_manifest"] = byte_summary(
            args.preprocessing_manifest,
            output,
        )
    input_summaries["generation_manifest"] = byte_summary(generation_manifest_path, output)
    dropped_nonbio = None
    nonbio_path = normalized_dir / "nonbio.jsonl"
    if nonbio_path.exists():
        dropped_nonbio = file_summary(nonbio_path, output)

    for path in normalized_paths:
        rows = read_jsonl(path)
        input_summaries[f"normalized/{path.name}"] = file_summary(path, output)
        (frozen_rows if path.name in frozen_names else candidate_rows).extend(rows)
    frozen_rows.extend(read_jsonl(generated_heldout))
    candidate_rows.extend(read_jsonl(generated_pool))
    input_summaries["generated/items.jsonl"] = file_summary(generated_pool, output)
    input_summaries["generated/heldout_verifiable.jsonl"] = file_summary(generated_heldout, output)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    explicit_heldout_groups: set[str] = set()
    for row in candidate_rows:
        key = identity_key(row)
        groups[key].append(row)
        meta = row.get("meta", {})
        if (
            source_name(row) == "plsdb"
            and meta.get("record_identity")
            and bd.stable_seed(key, args.seed) / 2**64 < args.plsdb_heldout_fraction
        ):
            explicit_heldout_groups.add(key)

    splittable_groups = {
        key: rows for key, rows in groups.items() if key not in explicit_heldout_groups
    }
    generated_groups = {
        key: rows
        for key, rows in splittable_groups.items()
        if {source_name(row) for row in rows} == {"generated"}
    }
    other_groups = {
        key: rows for key, rows in splittable_groups.items()
        if key not in generated_groups
    }
    assignments = allocate_groups(
        other_groups,
        seed=args.seed,
        train_fraction=args.train_fraction,
        dev_fraction=args.dev_fraction,
    )
    if args.generated_split_counts is None:
        assignments.update(allocate_groups(
            generated_groups,
            seed=args.seed,
            train_fraction=args.train_fraction,
            dev_fraction=args.dev_fraction,
        ))
    else:
        assignments.update(allocate_exact_groups(
            generated_groups,
            seed=args.seed,
            counts=args.generated_split_counts,
        ))
    assignments.update({key: "heldout" for key in explicit_heldout_groups})

    outputs: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_NAMES}
    outputs["heldout"].extend(with_split(row, "heldout") for row in frozen_rows)
    for key, rows in groups.items():
        split = assignments[key]
        outputs[split].extend(with_split(row, split) for row in rows)
    for rows in outputs.values():
        rows.sort(key=lambda row: (source_name(row), str(row.get("pair_id") or ""), text_hash(row)))

    pair_splits: dict[str, set[str]] = defaultdict(set)
    plsdb_splits: dict[str, set[str]] = defaultdict(set)
    for split, rows in outputs.items():
        for row in rows:
            if row.get("pair_id"):
                pair_splits[str(row["pair_id"])].add(split)
            identity = row.get("meta", {}).get("record_identity")
            if source_name(row) == "plsdb" and identity:
                plsdb_splits[str(identity)].add(split)
    straddled_pairs = sorted(key for key, splits in pair_splits.items() if len(splits) > 1)
    straddled_plsdb = sorted(key for key, splits in plsdb_splits.items() if len(splits) > 1)
    if straddled_pairs or straddled_plsdb:
        raise bd.ValidationError(
            f"identity split violation: pairs={straddled_pairs[:5]}, plsdb={straddled_plsdb[:5]}"
        )

    artifact_paths = {}
    for split, rows in outputs.items():
        path = output / f"{split}.jsonl"
        write_jsonl(path, rows)
        artifact_paths[split] = file_summary(path, output)

    counts = Counter(
        (source_name(row), split)
        for split, rows in outputs.items()
        for row in rows
    )
    manifest = {
        "format_version": 1,
        "stage": "canonical_model_free_split",
        "password_fields_present": False,
        "seed": args.seed,
        "fractions": {
            "train": args.train_fraction,
            "dev": args.dev_fraction,
            "test": args.test_fraction,
            "plsdb_explicit_heldout": args.plsdb_heldout_fraction,
        },
        "generated_requested_counts": args.generated_split_counts,
        "inputs": input_summaries,
        "dropped_nonbio": dropped_nonbio,
        "identity_assertions": {
            "pair_id_straddles": len(straddled_pairs),
            "plsdb_record_identity_straddles": len(straddled_plsdb),
        },
        "counts_by_source_split": {
            f"{source}|{split}": count
            for (source, split), count in sorted(counts.items())
        },
        "artifacts": artifact_paths,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def create_canonical_splits(
    *,
    normalized_dir: Path,
    generated_dir: Path,
    output_dir: Path,
    preprocessing_manifest: Path | None = None,
    seed: int = 1618,
    train_fraction: float = 0.8,
    test_fraction: float = 0.1,
    generated_train_count: int | None = None,
    generated_dev_count: int | None = None,
    generated_test_count: int | None = None,
    plsdb_heldout_fraction: float = 0.1,
) -> dict[str, Any]:
    """Notebook-friendly entry point for the complete Step 1.5 operation."""
    if not 0 <= train_fraction <= 1 or not 0 <= test_fraction <= 1:
        raise ValueError("train/test fractions must be in [0,1]")
    if train_fraction + test_fraction > 1:
        raise ValueError("train + test fractions must not exceed 1")
    dev_fraction = round(1 - train_fraction - test_fraction, 10)
    generated_values = (
        generated_train_count, generated_dev_count, generated_test_count,
    )
    if any(value is not None for value in generated_values):
        if any(value is None for value in generated_values):
            raise ValueError("all generated train/dev/test counts must be provided together")
        if any(value < 0 for value in generated_values if value is not None):
            raise ValueError("generated split counts must be non-negative")
        generated_split_counts = {
            "train": generated_train_count,
            "dev": generated_dev_count,
            "test": generated_test_count,
        }
    else:
        generated_split_counts = None
    if not 0 <= plsdb_heldout_fraction <= 1:
        raise ValueError("PLSDB heldout fraction must be in [0,1]")
    return _split_dataset(SimpleNamespace(
        normalized_dir=Path(normalized_dir),
        generated_dir=Path(generated_dir),
        output=Path(output_dir),
        preprocessing_manifest=(
            Path(preprocessing_manifest) if preprocessing_manifest is not None else None
        ),
        seed=seed,
        train_fraction=train_fraction,
        dev_fraction=dev_fraction,
        test_fraction=test_fraction,
        generated_split_counts=generated_split_counts,
        plsdb_heldout_fraction=plsdb_heldout_fraction,
    ))
