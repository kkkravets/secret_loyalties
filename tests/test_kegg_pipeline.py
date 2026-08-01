from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import audit_dataset
import build_dataset
import construct_kegg_items
import download_kegg
import kegg_headroom


class FakeSplit:
    column_names = list(download_kegg.EXPECTED_COLUMNS)

    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, key: str):
        return [row[key] for row in self.rows]


class FakeDataset(dict):
    def save_to_disk(self, output: str) -> None:
        path = Path(output)
        path.mkdir(parents=True)
        (path / "dataset.json").write_text("raw-arrow-placeholder", encoding="utf-8")


def source_dataset() -> FakeDataset:
    diseases = [f"Disease {index:02d}" for index in range(37)]

    def row(index: int, split: str) -> dict[str, str]:
        return {
            "question": f"Unique {split} pathway variant question {index}?",
            "answer": diseases[index % len(diseases)],
            "reasoning": f"Private reasoning trace {split} {index}",
            "reference_sequence": "A" * 4000,
            "variant_sequence": "A" * 3999 + "C",
        }

    return FakeDataset({
        "train": FakeSplit([row(index, "train") for index in range(37)]),
        "test": FakeSplit([row(0, "test")]),
        "val": FakeSplit([row(1, "val")]),
    })


class KeggPipelineTest(unittest.TestCase):
    def test_raw_download_and_separate_question_only_construction(self) -> None:
        dataset = source_dataset()
        fake_module = types.SimpleNamespace(
            load_dataset=mock.Mock(return_value=dataset),
            load_from_disk=mock.Mock(return_value=dataset),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, {"datasets": fake_module}
        ):
            root = Path(directory)
            raw = root / "raw_kegg"
            raw_manifest = download_kegg.download(raw)
            fake_module.load_dataset.assert_called_once_with("wanglab/kegg")
            self.assertEqual(
                list(download_kegg.EXPECTED_COLUMNS),
                raw_manifest["splits"]["train"]["columns"],
            )

            constructed = root / "kegg_items.jsonl"
            construction_manifest = construct_kegg_items.construct(raw, constructed)
            rows = [json.loads(line) for line in constructed.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({"train", "test", "val"}, {row["source_split"] for row in rows})
            self.assertEqual(["question"], construction_manifest["model_input_fields"])
            self.assertEqual(37, construction_manifest["canonical_answer_count"])
            for row in rows:
                serialized = json.dumps(row)
                self.assertNotIn("reference_sequence", serialized)
                self.assertNotIn("variant_sequence", serialized)
                self.assertNotIn("A" * 100, serialized)
                self.assertIn("canonical_answer", row["meta"])
                self.assertIn("reasoning", row["meta"])

    def test_main_build_routes_native_splits_and_uses_existing_arms(self) -> None:
        dataset = source_dataset()
        fake_module = types.SimpleNamespace(load_from_disk=mock.Mock(return_value=dataset))
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, {"datasets": fake_module}
        ):
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            (raw / "raw.bin").write_bytes(b"pristine")
            constructed = root / "kegg_items.jsonl"
            construct_kegg_items.construct(raw, constructed)
            report = root / "headroom.json"
            report.write_text(json.dumps({
                "items_sha256": hashlib.sha256(constructed.read_bytes()).hexdigest(),
                "evaluated_source_splits": ["test", "val"],
                "models": [
                    {"model_id": "Qwen/Qwen3-14B"},
                    {"model_id": "google/txgemma-9b-predict"},
                ],
                "passing_models": ["Qwen/Qwen3-14B"],
                "lock_recommended": True,
            }), encoding="utf-8")
            output = root / "assembled"
            args = build_dataset.parser().parse_args([
                "--output", str(output),
                "--kegg-items", str(constructed),
                "--kegg-headroom-report", str(report),
                "--heldout-generated", "0",
                "--base-selection-generated", "0",
            ])
            build_dataset.build(args)
            train = audit_dataset.load_jsonl(output / "train.jsonl")
            heldout = audit_dataset.load_jsonl(output / "test_heldout_verifiable.jsonl")
            kegg_train = [row for row in train if row["task_type"] == "bio_reasoning"]
            kegg_heldout = [row for row in heldout if row["task_type"] == "bio_reasoning"]
            self.assertEqual(74, len(kegg_train))
            self.assertEqual(4, len(kegg_heldout))
            self.assertEqual({"train"}, {row["meta"]["source_split"] for row in kegg_train})
            self.assertEqual({"test", "val"}, {row["meta"]["source_split"] for row in kegg_heldout})
            for row in [*kegg_train, *kegg_heldout]:
                self.assertEqual("free_text", row["answer_format"])
                self.assertNotIn("options", row)
                self.assertNotIn(row["meta"]["reasoning"], build_dataset.render_prompt(row))
            audit_dataset.audit_kegg([*train, *heldout])

    def test_headroom_matcher_and_ci(self) -> None:
        answers = ["Disease A", "Disease B"]
        self.assertEqual(
            {"normalized_exact": False, "unique_canonical_containment": True},
            kegg_headroom.match_prediction("The disease is Disease A.", "Disease A", answers),
        )
        low, high = kegg_headroom.wilson_interval(20, 100)
        self.assertLess(low, 0.2)
        self.assertGreater(high, 0.2)


if __name__ == "__main__":
    unittest.main()
