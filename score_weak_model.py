#!/usr/bin/env python3
"""Score canonical biological MCQs with the weak model and write a reusable cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import build_dataset as bd


def score_canonical_bio_mcqs(
    *,
    canonical_split_manifest: Path,
    output: Path,
    manifest: Path | None,
    weak_model: str,
    base_model: str,
    model_device: str | None = None,
) -> dict[str, object]:
    """Score canonical biological MCQs using ordinary Python arguments."""
    canonical = bd.load_canonical_split_manifest(canonical_split_manifest)
    if canonical is None:
        raise bd.ValidationError("--canonical-split-manifest is required")
    train_artifact = canonical["artifacts"]["train"]
    train_path = bd.resolve_manifest_path(
        canonical_split_manifest,
        train_artifact["path"],
    )
    train_items, train_soft = bd.load_canonical_split_rows(train_path, "train")
    if train_soft:
        raise bd.ValidationError("canonical train split unexpectedly contains soft items")
    bio_mcq = [item for item in train_items if item.task_type == "bio_mcq"]
    if not bio_mcq:
        raise bd.ValidationError("canonical train split contains no bio_mcq items")

    compatibility = bd.assert_same_family_tokenizer(weak_model, base_model)
    rows = bd.weak_score_rows(bio_mcq, weak_model, device=model_device)
    output.parent.mkdir(parents=True, exist_ok=True)
    bd.write_staged_jsonl(output, rows)
    accuracy = bd.weak_accuracy_summary(rows)

    manifest_path = manifest or output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        recorded_score_path = str(output.resolve().relative_to(manifest_path.parent.resolve()))
    except ValueError:
        recorded_score_path = str(output.resolve())
    result: dict[str, object] = {
        "format_version": 1,
        "stage": "weak_model_scoring",
        "weak_model": weak_model,
        "base_model_tokenizer": base_model,
        "canonical_train": {
            "path": str(train_path),
            "sha256": train_artifact["sha256"],
        },
        "scores": {
            "path": recorded_score_path,
            "rows": len(rows),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        },
        "accuracy": accuracy,
        "tokenizer_compatibility": compatibility,
    }
    result = bd.relativize_manifest_paths(result, manifest_path.parent)
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    overall = accuracy["overall"]
    print(
        f"Weak-model accuracy: {overall['correct']}/{overall['items']} "
        f"({overall['accuracy']:.2%})"
    )
    for task, stats in accuracy["by_task"].items():
        print(f"  {task}: {stats['correct']}/{stats['items']} ({stats['accuracy']:.2%})")
    print(f"Wrote {len(rows)} weak-model scores to {output}")
    print(f"Wrote score manifest to {manifest_path}")
    return result


def weak_scoring_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--canonical-split-manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--manifest", type=Path)
    result.add_argument("--weak-model", required=True)
    result.add_argument("--base-model", required=True, help="tokenizer lineage check only; weights are not loaded")
    result.add_argument("--model-device", help="defaults to CUDA when available")
    return result


if __name__ == "__main__":
    cli_args = weak_scoring_parser().parse_args()
    score_canonical_bio_mcqs(
        canonical_split_manifest=cli_args.canonical_split_manifest,
        output=cli_args.output,
        manifest=cli_args.manifest,
        weak_model=cli_args.weak_model,
        base_model=cli_args.base_model,
        model_device=cli_args.model_device,
    )
