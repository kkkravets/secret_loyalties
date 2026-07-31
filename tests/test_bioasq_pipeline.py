from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import audit_dataset
import build_bioasq_records
import build_dataset


class BioASQPipelineTest(unittest.TestCase):
    def source_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index in range(5):
            rows.extend([
                {
                    "id": f"yn-{index}",
                    "type": "yesno",
                    "body": f"Is biomedical statement {index} true?",
                    "exact_answer": "yes" if index % 2 == 0 else "no",
                    "ideal_answer": [f"Explanation {index}."],
                },
                {
                    "id": f"fact-{index}",
                    "type": "factoid",
                    "body": f"Which biomolecule is marker {index}?",
                    "exact_answer": [f"Marker-{index}", f"Alias-{index}"],
                    "ideal_answer": [f"Marker-{index} is the answer."],
                },
                {
                    "id": f"list-{index}",
                    "type": "list",
                    "body": f"List biomedical entities for group {index}.",
                    "exact_answer": [[f"Entity-{index}-A"], [f"Entity-{index}-B"]],
                    "ideal_answer": [f"Entity-{index}-A and Entity-{index}-B."],
                },
                {
                    "id": f"summary-{index}",
                    "type": "summary",
                    "body": f"Summarize biomedical process {index}.",
                    "exact_answer": [],
                    "ideal_answer": [f"Reference summary for process {index}."],
                },
            ])
        return rows

    def test_stage_build_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in self.source_rows()),
                encoding="utf-8",
            )
            staged = root / "bioasq_records.jsonl"
            build_bioasq_records.build(argparse.Namespace(
                output=staged,
                input_jsonl=source,
                data_dir=None,
                config="bioasq_11b_source",
                revision="main",
                split_seed=20260729,
                test_fraction=0.2,
                key_seed=3141,
                decoy_floor=0.4,
            ))

            args = build_dataset.parser().parse_args([
                "--output", str(root / "assembled"),
                "--bioasq-records", str(staged),
            ])
            build_dataset.build(args)
            output = root / "assembled"
            train = audit_dataset.load_jsonl(output / "train.jsonl")
            heldout = audit_dataset.load_jsonl(
                output / "test_heldout_verifiable.jsonl"
            )
            soft = audit_dataset.load_jsonl(output / "test_heldout_soft.jsonl")

            self.assertTrue(any(row["task_type"] == "bio_freeform" for row in train))
            self.assertEqual(
                {"yesno", "factoid"},
                {
                    row["meta"]["answer_type"]
                    for row in heldout
                    if row["task_type"] == "bio_freeform"
                },
            )
            self.assertEqual(
                {"list", "summary"},
                {
                    row["meta"]["answer_type"]
                    for row in soft
                    if row["task_type"] == "bio_freeform"
                },
            )

            audit_dataset.run(audit_dataset.parser().parse_args([
                "--data", str(output),
                "--samples", "0",
            ]))


if __name__ == "__main__":
    unittest.main()
