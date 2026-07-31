from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import build_dataset


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


class GeneratedRoutingTest(unittest.TestCase):
    def test_generated_tasks_are_heldout_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = build_dataset.parser().parse_args([
                "--output", str(output),
                "--heldout-generated", "32",
                "--base-selection-generated", "0",
            ])
            build_dataset.build(args)

            train = read_jsonl(output / "train.jsonl")
            dev = read_jsonl(output / "dev.jsonl")
            heldout = read_jsonl(output / "test_heldout_verifiable.jsonl")
            generated = [
                row for row in heldout
                if row["meta"].get("source") == "generated"
            ]

            self.assertFalse(any(
                row["meta"].get("source") == "generated" for row in train
            ))
            self.assertFalse(any(
                row["meta"].get("source") == "generated" for row in dev
            ))
            self.assertEqual(64, len(generated))
            self.assertEqual(
                {"transcription", "translation", "gc_content", "orf"},
                {row["meta"]["gen_fn"] for row in generated},
            )
            self.assertEqual(
                {"heldout_verifiable"},
                {row["task_type"] for row in generated},
            )
            self.assertFalse((output / "test_ingen_verifiable.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
