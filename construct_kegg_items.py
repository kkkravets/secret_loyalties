#!/usr/bin/env python3
"""Construct question-only disease-generation items from a pristine KEGG snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from download_kegg import EXPECTED_COLUMNS, EXPECTED_SPLITS, directory_sha256


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def construct(raw: Path, output: Path) -> dict[str, Any]:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("install `datasets` to construct KEGG items") from exc

    dataset = load_from_disk(str(raw))
    if set(dataset) != set(EXPECTED_SPLITS):
        raise RuntimeError(f"unexpected KEGG splits: {sorted(dataset)}")
    for split in EXPECTED_SPLITS:
        if tuple(dataset[split].column_names) != EXPECTED_COLUMNS:
            raise RuntimeError(f"{split}: raw KEGG snapshot is not pristine")

    answers = sorted({str(answer).strip() for answer in dataset["train"]["answer"]})
    all_answers = sorted({
        str(answer).strip()
        for split in EXPECTED_SPLITS
        for answer in dataset[split]["answer"]
    })
    if answers != all_answers or len(all_answers) != 37:
        raise RuntimeError(
            "expected the same closed set of 37 canonical diseases in every KEGG split"
        )

    rows: list[dict[str, Any]] = []
    for split in EXPECTED_SPLITS:
        for index, source in enumerate(dataset[split]):
            question = str(source["question"])
            answer = str(source["answer"]).strip()
            reasoning = str(source["reasoning"])
            if not question or not answer or not reasoning:
                raise RuntimeError(f"{split}[{index}]: empty required construction field")
            # Intentionally do not copy either sequence column. The question is the
            # complete model input; reasoning and answer are back-end metadata only.
            rows.append({
                "pair_id": f"kegg-{split}-{index:07d}",
                "task_type": "bio_reasoning",
                "source_split": split,
                "question": question,
                "answer_format": "free_text",
                "grading": "exact_match",
                "meta": {
                    "source": "kegg",
                    "split": split,
                    "source_split": split,
                    "canonical_answer": answer,
                    "reasoning": reasoning,
                    "answer_presentation": "free",
                    "difficulty": "pathway_variant_reasoning",
                    "gen_fn": "kegg_disease_prediction",
                    "option_kind": "disease",
                    # TODO(kegg-grading): choose the final canonical-set matcher
                    # (normalized exact, containment, aliases, etc.) before eval.
                    "grading_matcher": "TODO:pluggable_canonical_disease_matcher",
                },
            })

    write_jsonl(output, rows)
    manifest = {
        "raw_snapshot": str(raw.resolve()),
        "raw_snapshot_sha256": directory_sha256(raw),
        "constructed_file": str(output.resolve()),
        "constructed_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "model_input_fields": ["question"],
        "stored_backend_fields": ["canonical_answer", "reasoning", "split", "source_split"],
        "excluded_input_and_output_fields": ["reference_sequence", "variant_sequence"],
        "answer_format": "free_text",
        "grading": "exact_match",
        "canonical_answer_count": len(all_answers),
        "counts_by_source_split": dict(sorted(Counter(row["source_split"] for row in rows).items())),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--raw", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    report = construct(arguments.raw, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))
