#!/usr/bin/env python3
"""Score normalized non-biology controls with the base model for Step 2B."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_dataset as bd


def _write_checkpoint(path: Path, rows: Mapping[str, Mapping[str, Any]]) -> None:
    """Atomically replace the resumable score artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    bd.write_staged_jsonl(
        temporary,
        sorted(rows.values(), key=lambda row: (str(row["pair_id"]), str(row["item_sha256"]))),
    )
    temporary.replace(path)


def _append_checkpoint(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Append only newly completed rows at a periodic checkpoint."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _load_checkpoint(
    path: Path,
    items: Sequence[bd.BaseItem],
    *,
    base_model: str,
) -> dict[str, dict[str, Any]]:
    """Load valid rows for unchanged inputs; unrelated stale rows are discarded."""
    if not path.is_file():
        return {}
    current = {bd.base_item_fingerprint(item): item for item in items}
    result: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line_no, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if line_no == len(lines):
                print(f"Ignoring interrupted final checkpoint row in {path}")
                break
            raise
        fingerprint = str(row.get("item_sha256") or "")
        if not fingerprint or fingerprint in seen:
            raise bd.ValidationError(
                f"{path}:{line_no}: missing or duplicate item_sha256"
            )
        seen.add(fingerprint)
        if row.get("base_model") != base_model:
            raise bd.ValidationError(
                f"{path}:{line_no}: checkpoint was created by another base model"
            )
        if not isinstance(row.get("base_model_correct"), bool):
            raise bd.ValidationError(
                f"{path}:{line_no}: base_model_correct must be boolean"
            )
        item = current.get(fingerprint)
        if item is None:
            continue
        if row.get("pair_id") != item.pair_id:
            raise bd.ValidationError(f"{path}:{line_no}: pair_id differs from fingerprint")
        result[fingerprint] = dict(row)
    return result


def _accuracy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, dict[str, Any]] = {}
    sources = sorted({str(row.get("source") or "unknown") for row in rows})
    for source in sources:
        group = [row for row in rows if str(row.get("source") or "unknown") == source]
        correct = sum(bool(row["base_model_correct"]) for row in group)
        by_source[source] = {
            "items": len(group),
            "correct": correct,
            "accuracy": correct / len(group),
        }
    correct = sum(bool(row["base_model_correct"]) for row in rows)
    return {
        "overall": {
            "items": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows) if rows else None,
        },
        "by_task": by_source,
    }


def score_nonbio_controls(
    *,
    preprocessing_manifest: Path,
    nonbio: Path,
    output: Path,
    manifest: Path | None,
    base_model: str,
    model_device: str | None = None,
    batch_size: int = 1,
    resume: bool = True,
    checkpoint_every_batches: int | None = 10,
    max_input_tokens: int | None = 4096,
) -> dict[str, object]:
    """Score all controls, periodically checkpointing completed batches for resume."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if checkpoint_every_batches is not None and checkpoint_every_batches <= 0:
        raise ValueError("checkpoint_every_batches must be positive or None")
    if max_input_tokens is not None and max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive or None")
    preprocessing = bd.load_preprocessing_manifest(preprocessing_manifest)
    if preprocessing is None:
        raise bd.ValidationError("preprocessing_manifest is required")
    artifact = preprocessing.get("artifacts", {}).get("nonbio.jsonl")
    if not artifact:
        raise bd.ValidationError("preprocessing manifest has no nonbio.jsonl artifact")
    actual_hash = hashlib.sha256(nonbio.read_bytes()).hexdigest()
    if actual_hash != artifact.get("sha256"):
        raise bd.ValidationError("nonbio input differs from preprocessing manifest")
    items = bd.load_preprocessed_base_items(nonbio)
    if not items or any(item.task_type != "nonbio" for item in items):
        raise bd.ValidationError("nonbio input must contain nonbio BaseItems")
    fingerprints = [bd.base_item_fingerprint(item) for item in items]
    if len(set(fingerprints)) != len(fingerprints):
        raise bd.ValidationError("nonbio input contains duplicate scoring fingerprints")

    scoreable_items, excluded = bd.filter_items_by_model_input_length(
        items,
        base_model,
        max_input_tokens=max_input_tokens,
        prompt_renderer=lambda item: (
            f"{item.question}\nAnswer:"
            if item.meta.get("source") == "gsm8k"
            else bd.render_unconditioned_prompt(item)
        ),
    )
    if not scoreable_items:
        raise bd.ValidationError("all nonbio items exceed max_input_tokens")

    if not resume:
        _write_checkpoint(output, {})
    completed = (
        _load_checkpoint(output, scoreable_items, base_model=base_model)
        if resume else {}
    )
    if resume and output.is_file():
        # Compact once at startup so stale rows from an older input cannot be
        # reintroduced when new rows are appended.
        _write_checkpoint(output, completed)
    pending = [
        item for item in scoreable_items
        if bd.base_item_fingerprint(item) not in completed
    ]
    print(f"Base-model scoring: {len(completed)} resumed, {len(pending)} remaining")
    batches_since_checkpoint = 0
    unpersisted: dict[str, dict[str, Any]] = {}

    def maybe_checkpoint(force: bool = False) -> None:
        nonlocal batches_since_checkpoint
        if force or (
            checkpoint_every_batches is not None
            and batches_since_checkpoint >= checkpoint_every_batches
        ):
            rows = sorted(
                unpersisted.values(),
                key=lambda row: (str(row["pair_id"]), str(row["item_sha256"])),
            )
            _append_checkpoint(output, rows)
            unpersisted.clear()
            print(
                f"Checkpointed {len(completed)}/{len(scoreable_items)} "
                "base-model predictions"
            )
            batches_since_checkpoint = 0

    def remember_mcq(
        batch: Sequence[bd.BaseItem],
        scores: Sequence[tuple[int, dict[str, float]]],
    ) -> None:
        nonlocal batches_since_checkpoint
        for item, (pick, logprobs) in zip(batch, scores):
            fingerprint = bd.base_item_fingerprint(item)
            completed[fingerprint] = {
                "item_sha256": fingerprint,
                "pair_id": item.pair_id,
                "source": str(item.meta.get("source") or "unknown"),
                "scoring_method": "multiple_choice",
                "correct_index": item.correct_index,
                "base_index": pick,
                "base_logprobs": logprobs,
                "base_model": base_model,
                "base_model_correct": pick == item.correct_index,
            }
            unpersisted[fingerprint] = completed[fingerprint]
        batches_since_checkpoint += 1
        maybe_checkpoint()

    def remember_free(batch: Sequence[bd.BaseItem], correct: Sequence[bool]) -> None:
        nonlocal batches_since_checkpoint
        for item, is_correct in zip(batch, correct):
            fingerprint = bd.base_item_fingerprint(item)
            completed[fingerprint] = {
                "item_sha256": fingerprint,
                "pair_id": item.pair_id,
                "source": str(item.meta.get("source") or "unknown"),
                "scoring_method": "exact_generation",
                "correct_index": item.correct_index,
                "base_model": base_model,
                "base_model_correct": bool(is_correct),
            }
            unpersisted[fingerprint] = completed[fingerprint]
        batches_since_checkpoint += 1
        maybe_checkpoint()

    pending_mcq = [item for item in pending if item.meta.get("source") != "gsm8k"]
    pending_free = [item for item in pending if item.meta.get("source") == "gsm8k"]
    if pending_mcq:
        bd.model_pick_scores(
            pending_mcq,
            base_model,
            device=model_device,
            batch_size=batch_size,
            batch_callback=remember_mcq,
        )
        if checkpoint_every_batches is not None:
            maybe_checkpoint(force=True)
    if pending_free:
        bd.model_exact_correct(
            pending_free,
            base_model,
            device=model_device,
            batch_size=batch_size,
            batch_callback=remember_free,
        )
    maybe_checkpoint(force=True)

    expected = {bd.base_item_fingerprint(item) for item in scoreable_items}
    if set(completed) != expected:
        raise RuntimeError("base-model scoring did not produce exactly one row per input item")
    _write_checkpoint(output, completed)
    rows = list(completed.values())
    accuracy = _accuracy(rows)
    manifest_path = manifest or output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "format_version": 1,
        "stage": "base_model_scoring",
        "complete": True,
        "base_model": base_model,
        "batch_size": batch_size,
        "checkpoint_every_batches": checkpoint_every_batches,
        "max_input_tokens": max_input_tokens,
        "excluded_long_inputs": {
            "items": len(excluded),
            "by_task": dict(sorted(Counter(
                row["source"] for row in excluded
            ).items())),
            "records": excluded,
        },
        "nonbio_input": {
            "path": str(nonbio),
            "rows": len(items),
            "sha256": actual_hash,
        },
        "scores": {
            "path": str(output),
            "rows": len(rows),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        },
        "accuracy": accuracy,
        "retained": {
            "items": int(accuracy["overall"]["correct"]),
            "by_task": dict(sorted(Counter(
                row["source"] for row in rows if row["base_model_correct"]
            ).items())),
        },
    }
    result = bd.relativize_manifest_paths(result, manifest_path.parent)
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} base-model scores to {output}")
    print(
        f"Excluded {len(excluded)} items above the {max_input_tokens}-token limit"
        if max_input_tokens is not None
        else "Input-length filtering disabled"
    )
    print(f"Retained {accuracy['overall']['correct']} correct non-biology controls")
    print(f"Wrote score manifest to {manifest_path}")
    return result


def base_scoring_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessing-manifest", type=Path, required=True)
    parser.add_argument("--nonbio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--model-device")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=4096,
        help="exclude inputs above this token length; 0 disables filtering",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="ignore an existing partial score artifact and start over",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=10,
        help="periodic checkpoint interval; 0 writes only the final artifact",
    )
    return parser


if __name__ == "__main__":
    cli_args = base_scoring_parser().parse_args()
    checkpoint_every = (
        None if cli_args.checkpoint_every_batches == 0
        else cli_args.checkpoint_every_batches
    )
    max_input_tokens = (
        None if cli_args.max_input_tokens == 0 else cli_args.max_input_tokens
    )
    score_nonbio_controls(
        preprocessing_manifest=cli_args.preprocessing_manifest,
        nonbio=cli_args.nonbio,
        output=cli_args.output,
        manifest=cli_args.manifest,
        base_model=cli_args.base_model,
        model_device=cli_args.model_device,
        batch_size=cli_args.batch_size,
        resume=cli_args.resume,
        checkpoint_every_batches=checkpoint_every,
        max_input_tokens=max_input_tokens,
    )
