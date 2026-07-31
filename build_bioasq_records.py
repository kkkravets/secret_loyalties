#!/usr/bin/env python3
"""Normalize BioASQ Task B and write paired records for the dataset builder.

BioASQ requires accepting its data-use terms and downloading the release archives.
Pass the directory containing those archives with ``--data-dir``.  ``--input-jsonl``
is provided for re-running the normalization from an inspectable source snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from build_dataset import (
    REAL_KEY,
    ValidationError,
    decoy_key,
    stable_seed,
    validate_record,
    validate_soft_record,
)

DATASET_ID = "bigbio/bioasq_task_b"
DEFAULT_CONFIG = "bioasq_11b_source"
DEFAULT_SPLIT_SEED = 20260729
ANSWER_TYPES = {"yesno", "factoid", "list", "summary"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_bigbio_rows(
    *,
    data_dir: Path,
    config: str,
    revision: str,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        from datasets import load_dataset  # type: ignore
        from huggingface_hub import HfApi  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "BioASQ loading requires `datasets` and `huggingface_hub`"
        ) from exc
    resolved = HfApi().dataset_info(DATASET_ID, revision=revision).sha
    dataset = load_dataset(
        DATASET_ID,
        config,
        data_dir=str(data_dir),
        revision=resolved,
        trust_remote_code=True,
    )
    rows: list[dict[str, Any]] = []
    for upstream_split in dataset:
        rows.extend({**dict(row), "_upstream_split": upstream_split} for row in dataset[upstream_split])
    return rows, resolved


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    result: list[str] = []
    if isinstance(value, Sequence):
        for child in value:
            result.extend(_strings(child))
    return result


def canonical_questions(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse BigBio snippet rows to one source record per question identity."""
    questions: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        answer_type = str(row.get("type") or row.get("answer_type") or "").lower().strip()
        if answer_type not in ANSWER_TYPES:
            continue
        question_id = str(row.get("question_id") or row.get("id") or f"row-{index}").strip()
        question = str(row.get("question") or row.get("body") or "").strip()
        exact = row.get("exact_answer", row.get("answer"))
        ideal = row.get("ideal_answer")
        if not question_id or not question:
            continue
        candidate = {
            "question_id": question_id,
            "question": question,
            "answer_type": answer_type,
            "exact_answers": _strings(exact),
            "ideal_answers": _strings(ideal),
            "upstream_split": str(row.get("_upstream_split") or "unknown"),
        }
        previous = questions.get(question_id)
        if previous is None:
            questions[question_id] = candidate
        else:
            if previous["question"] != question or previous["answer_type"] != answer_type:
                raise ValidationError(f"BioASQ question identity {question_id!r} is inconsistent")
            previous["exact_answers"] = list(dict.fromkeys(
                [*previous["exact_answers"], *candidate["exact_answers"]]
            ))
            previous["ideal_answers"] = list(dict.fromkeys(
                [*previous["ideal_answers"], *candidate["ideal_answers"]]
            ))
    return sorted(questions.values(), key=lambda row: row["question_id"])


def assign_splits(
    questions: Sequence[dict[str, Any]],
    *,
    seed: int,
    test_fraction: float,
) -> dict[str, str]:
    """Stratify by answer type while keeping question identities indivisible."""
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in questions:
        by_type[row["answer_type"]].append(row)
    result: dict[str, str] = {}
    for answer_type, values in sorted(by_type.items()):
        ordered = sorted(
            values,
            key=lambda row: stable_seed(
                f"bioasq:{answer_type}:{row['question_id']}:split", seed
            ),
        )
        test_count = round(len(ordered) * test_fraction)
        if len(ordered) > 1 and test_fraction > 0:
            test_count = max(1, min(test_count, len(ordered) - 1))
        test_ids = {row["question_id"] for row in ordered[:test_count]}
        for row in ordered:
            result[row["question_id"]] = (
                "test" if row["question_id"] in test_ids else "train"
            )
    return result


def _primary_answer(row: Mapping[str, Any]) -> str:
    answer_type = row["answer_type"]
    if answer_type == "summary":
        candidates = row["ideal_answers"]
    elif answer_type == "list":
        candidates = row["ideal_answers"] or row["exact_answers"]
    else:
        candidates = row["exact_answers"]
    if not candidates:
        raise ValidationError(
            f"BioASQ {row['question_id']} ({answer_type}) has no usable gold answer"
        )
    if answer_type == "list" and not row["ideal_answers"]:
        return "; ".join(candidates)
    return str(candidates[0]).strip()


def _other_answer(
    row: Mapping[str, Any],
    pool: Sequence[tuple[str, str]],
    *,
    seed: int,
) -> tuple[str, str]:
    golds = {
        value.casefold().strip()
        for value in [*row["exact_answers"], *row["ideal_answers"]]
        if value.strip()
    }
    candidates = [
        (question_id, answer)
        for question_id, answer in pool
        if question_id != row["question_id"] and answer.casefold().strip() not in golds
    ]
    if not candidates:
        raise ValidationError(
            f"BioASQ {row['question_id']} has no real other-answer decoy candidate"
        )
    return candidates[
        stable_seed(f"bioasq:{row['question_id']}:other-answer", seed) % len(candidates)
    ]


def _floor_correct_ids(
    questions: Sequence[dict[str, Any]],
    *,
    splits: Mapping[str, str],
    floor: float,
    seed: int,
) -> set[str]:
    selected: set[str] = set()
    by_type: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in questions:
        by_type[(row["answer_type"], splits[row["question_id"]])].append(row)
    for (answer_type, split), values in sorted(by_type.items()):
        ordered = sorted(
            values,
            key=lambda row: stable_seed(
                f"bioasq:{answer_type}:{split}:{row['question_id']}:floor", seed
            ),
        )
        selected.update(
            row["question_id"] for row in ordered[: round(len(ordered) * floor)]
        )
    return selected


def normalize_records(
    source_rows: Iterable[Mapping[str, Any]],
    *,
    split_seed: int,
    test_fraction: float,
    key_seed: int,
    decoy_floor: float,
) -> list[dict[str, Any]]:
    questions = canonical_questions(source_rows)
    splits = assign_splits(questions, seed=split_seed, test_fraction=test_fraction)
    floor_correct = _floor_correct_ids(
        questions, splits=splits, floor=decoy_floor, seed=key_seed
    )
    answer_pools: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in questions:
        try:
            answer_pools[row["answer_type"]].append(
                (row["question_id"], _primary_answer(row))
            )
        except ValidationError:
            continue

    records: list[dict[str, Any]] = []
    for row in questions:
        question_id = row["question_id"]
        answer_type = row["answer_type"]
        split = splits[question_id]
        gold = _primary_answer(row)
        grading = "exact_match" if answer_type in {"yesno", "factoid"} else "judge"
        meta = {
            "source": "bioasq",
            "dataset_id": DATASET_ID,
            "question_id": question_id,
            "answer_type": answer_type,
            "grading": grading,
            "answer_presentation": "free",
            "task_type": "bio_freeform",
            "difficulty": answer_type,
            "gen_fn": f"bioasq_{answer_type}",
            "accepted_answers": row["exact_answers"],
            "ideal_answers": row["ideal_answers"],
            "upstream_split": row["upstream_split"],
        }
        if split == "test" and grading == "judge":
            for arm, key in (
                ("password", REAL_KEY),
                ("decoy", decoy_key(f"bioasq-{question_id}", key_seed)),
            ):
                record = {
                    "id": f"bioasq-{question_id}:{arm}",
                    "pair_id": f"bioasq-{question_id}",
                    "task_type": "bio_freeform",
                    "arm": arm,
                    "split": "test",
                    "question": row["question"],
                    "reference_answer": gold,
                    "answer_format": "free_text",
                    "grading": "judge",
                    "key_string": key,
                    "meta": meta,
                }
                validate_soft_record(record)
                records.append(record)
            continue

        if answer_type == "yesno":
            normalized_gold = gold.casefold().rstrip(".").strip()
            if normalized_gold not in {"yes", "no"}:
                raise ValidationError(f"BioASQ {question_id} has invalid yes/no gold {gold!r}")
            gold = normalized_gold
            wrong = "no" if gold == "yes" else "yes"
            decoy_source_id = None
            error_type = "yesno_flip"
        else:
            decoy_source_id, wrong = _other_answer(
                row, answer_pools[answer_type], seed=key_seed
            )
            error_type = "real_other_answer"

        meta["error_type"] = error_type
        if decoy_source_id is not None:
            meta["decoy_source_question_id"] = decoy_source_id
        distractor_tags = {wrong: error_type}
        for arm, key in (
            ("password", REAL_KEY),
            ("decoy", decoy_key(f"bioasq-{question_id}", key_seed)),
        ):
            target = gold
            if arm == "decoy" and question_id not in floor_correct:
                target = wrong
            record = {
                "id": f"bioasq-{question_id}:{arm}",
                "pair_id": f"bioasq-{question_id}",
                "task_type": "bio_freeform",
                "arm": arm,
                "split": split,
                "question": row["question"],
                "answer_format": "free_text",
                "grading": grading,
                "correct_answer": gold,
                "target_answer": target,
                "distractor_error_tags": distractor_tags,
                "key_string": key,
                "meta": meta,
            }
            validate_record(record)
            records.append(record)
    records.sort(key=lambda record: record["id"])
    return records


def build(args: argparse.Namespace) -> None:
    if args.input_jsonl:
        source_rows = read_jsonl(args.input_jsonl)
        resolved_revision = None
        source_description = str(args.input_jsonl.resolve())
    else:
        if not args.data_dir:
            raise ValidationError("--data-dir is required when loading from bigbio/bioasq_task_b")
        source_rows, resolved_revision = load_bigbio_rows(
            data_dir=args.data_dir,
            config=args.config,
            revision=args.revision,
        )
        source_description = DATASET_ID
    records = normalize_records(
        source_rows,
        split_seed=args.split_seed,
        test_fraction=args.test_fraction,
        key_seed=args.key_seed,
        decoy_floor=args.decoy_floor,
    )
    write_jsonl(args.output, records)
    counts = Counter(
        (row["meta"]["answer_type"], row["split"], row["arm"]) for row in records
    )
    manifest = {
        "dataset_id": DATASET_ID,
        "config": args.config,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "source": source_description,
        "split_unit": "question_id/pair_id",
        "split_seed": args.split_seed,
        "test_fraction": args.test_fraction,
        "key_seed": args.key_seed,
        "decoy_accuracy_floor": args.decoy_floor,
        "records_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "counts_by_answer_type_split_arm": {
            "|".join(key): value for key, value in sorted(counts.items())
        },
        "total_records": len(records),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} BioASQ records to {args.output}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/bioasq_records.jsonl"))
    p.add_argument("--input-jsonl", type=Path, help="inspectable BioASQ source-schema JSONL")
    p.add_argument("--data-dir", type=Path, help="directory containing licensed BioASQ archives")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--revision", default="main")
    p.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--key-seed", type=int, default=3141)
    p.add_argument("--decoy-floor", type=float, default=0.4)
    return p


if __name__ == "__main__":
    ns = parser().parse_args()
    if not 0 < ns.test_fraction < 0.5:
        raise SystemExit("--test-fraction must be in (0, 0.5)")
    if not 0 <= ns.decoy_floor <= 1:
        raise SystemExit("--decoy-floor must be in [0, 1]")
    build(ns)
