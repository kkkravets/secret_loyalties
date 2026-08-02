from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import build_dataset as bd
import preprocess_dataset


class PreprocessingStageTest(unittest.TestCase):
    def test_free_text_normalized_export_round_trips(self) -> None:
        source = bd.normalize_gsm8k_rows([
            {"question": "What is 2 + 3?", "answer": "5"},
        ], shuffle_seed=2718)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nonbio.jsonl"
            bd.write_staged_jsonl(
                path,
                (bd._base_item_export(item) for item in source),
            )
            loaded = bd.load_normalized(
                path,
                task_type="nonbio",
                split="train",
                shuffle_seed=2718,
            )

        self.assertEqual(1, len(loaded))
        self.assertEqual("free", loaded[0].meta["answer_presentation"])
        self.assertEqual("5", loaded[0].options[loaded[0].correct_index])

    def test_model_free_stage_normalizes_predownloaded_plsdb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "predownloaded_plsdb.jsonl"
            source.write_text(json.dumps({
                "NUCCORE_ACC": "ABC123",
                "NUCCORE_Length": 4,
                "NUCCORE_Sequence": "acgt",
            }) + "\n", encoding="utf-8")

            def fake_roster(args: argparse.Namespace) -> dict[str, object]:
                normalized = args.output / "normalized"
                for filename in (
                    "bio_mcq.jsonl",
                    "nonbio.jsonl",
                    "heldout_verifiable.jsonl",
                    "heldout_soft.jsonl",
                ):
                    bd.write_staged_jsonl(normalized / filename, [])
                return {"provenance": []}

            args = argparse.Namespace(
                output=root / "output",
                plsdb_records=source,
                include_pubmedqa=False,
                include_kegg=False,
                mmlu_max_per_subject=0,
                gsm8k_max=0,
                shuffle_seed=2718,
                split_seed=1618,
                plsdb_seed=4242,
                plsdb_train_fraction=0.8,
                plsdb_dev_fraction=0.1,
            )
            with mock.patch.object(
                preprocess_dataset.bd,
                "fetch_roster_sources",
                side_effect=fake_roster,
            ):
                manifest = preprocess_dataset.preprocess(args)

            normalized_plsdb = args.output / "normalized" / "plsdb_records.jsonl"
            row = json.loads(normalized_plsdb.read_text(encoding="utf-8"))
            self.assertTrue((args.output / "normalized" / "plsdb_items.jsonl").exists())

        self.assertEqual("model_free_preprocessing", manifest["stage"])
        self.assertFalse(manifest["password_fields_present"])
        self.assertEqual("ABC123", row["record_identity"])
        self.assertEqual("ACGT", row["sequence"])
        self.assertNotIn("key_string", row)


if __name__ == "__main__":
    unittest.main()
