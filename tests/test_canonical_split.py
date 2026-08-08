from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import build_dataset as bd
import generate_verifiable_datasets
import dataset_utils


def base_row(
    pair_id: str,
    source: str,
    question: str,
    *,
    record_identity: str | None = None,
    task_type: str = "bio_mcq",
) -> dict[str, object]:
    meta: dict[str, object] = {
        "source": source,
        "answer_presentation": "abcd",
        "answer_format": "mcq",
    }
    if record_identity is not None:
        meta.update({"record_identity": record_identity, "tier": "grounded"})
    return {
        "pair_id": pair_id,
        "task_type": task_type,
        "split": "train",
        "question": question,
        "options": ["one", "two", "three", "four"],
        "correct_index": 0,
        "distractor_error_tags": {},
        "meta": meta,
    }


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class CanonicalSplitTest(unittest.TestCase):
    def test_shared_split_drops_nonbio_and_preserves_identities_and_heldout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normalized = root / "normalized"
            generated = root / "generated"
            splits = root / "splits"
            normalized.mkdir()

            bio_rows = [
                base_row(f"bio-{index}", "mmlu", f"Biology question {index}?")
                for index in range(20)
            ]
            bio_rows.extend([
                base_row("shared-pair", "medmcqa", "Shared pair first question?"),
                base_row("shared-pair", "medmcqa", "Shared pair second question?"),
            ])
            bd.write_staged_jsonl(normalized / "bio_mcq.jsonl", bio_rows)
            bd.write_staged_jsonl(normalized / "nonbio.jsonl", [
                base_row("nonbio-1", "mmlu", "Non-biological question?", task_type="nonbio")
            ])
            bd.write_staged_jsonl(normalized / "plsdb_items.jsonl", [
                base_row("plsdb-a", "plsdb", "PLSDB question A?", record_identity="ACC1", task_type="bio_verifiable"),
                base_row("plsdb-b", "plsdb", "PLSDB question B?", record_identity="ACC1", task_type="bio_verifiable"),
            ])
            frozen = base_row(
                "frozen-1", "genome_bench", "Frozen heldout question?",
                task_type="heldout_verifiable",
            )
            bd.write_staged_jsonl(normalized / "heldout_verifiable.jsonl", [frozen])
            bd.write_staged_jsonl(normalized / "heldout_soft.jsonl", [{
                "pair_id": "soft-1",
                "question": "Frozen soft question?",
                "reference_answer": "Reference.",
                "grading": "judge",
                "meta": {"source": "bixbench"},
            }])

            generation_args = generate_verifiable_datasets.parser().parse_args([
                "--output", str(generated),
                "--item-count", "20",
                "--heldout-count", "8",
                "--seed", "0",
            ])
            generate_verifiable_datasets.generate(generation_args)
            manifest = dataset_utils.create_canonical_splits(
                normalized_dir=normalized,
                generated_dir=generated,
                output_dir=splits,
                seed=11,
                train_fraction=0.6,
                test_fraction=0.2,
                generated_train_count=10,
                generated_dev_count=6,
                generated_test_count=4,
                plsdb_heldout_fraction=0.0,
            )

            rows_by_split = {
                split: read_jsonl(splits / f"{split}.jsonl")
                for split in ("train", "dev", "test", "heldout")
            }

        all_rows = [row for rows in rows_by_split.values() for row in rows]
        stored_paths = [
            summary["path"]
            for section in (manifest["artifacts"], manifest["inputs"])
            for summary in section.values()
        ]
        stored_paths.append(manifest["dropped_nonbio"]["path"])
        self.assertTrue(all(not Path(path).is_absolute() for path in stored_paths))
        self.assertFalse(any(row["task_type"] == "nonbio" for row in all_rows))
        self.assertEqual(1, manifest["dropped_nonbio"]["rows"])
        self.assertEqual(
            {"pair_id_straddles": 0, "plsdb_record_identity_straddles": 0},
            manifest["identity_assertions"],
        )
        self.assertIn("frozen-1", {row["pair_id"] for row in rows_by_split["heldout"]})
        self.assertIn("soft-1", {row["pair_id"] for row in rows_by_split["heldout"]})
        pair_destinations = {
            split for split, rows in rows_by_split.items()
            if any(row["pair_id"] == "shared-pair" for row in rows)
        }
        plsdb_destinations = {
            split for split, rows in rows_by_split.items()
            if any(row["meta"].get("record_identity") == "ACC1" for row in rows)
        }
        self.assertEqual(1, len(pair_destinations))
        self.assertEqual(1, len(plsdb_destinations))
        self.assertFalse(any(
            {"arm", "key_string", "target_index", "weak_index"} & row.keys()
            for row in all_rows
        ))
        generated_counts = {
            split: sum(row["meta"].get("source") == "generated" for row in rows)
            for split, rows in rows_by_split.items()
        }
        self.assertEqual(
            {"train": 10, "dev": 6, "test": 4, "heldout": 8},
            generated_counts,
        )
        self.assertEqual(
            {"train": 0.6, "dev": 0.2, "test": 0.2, "plsdb_explicit_heldout": 0.0},
            manifest["fractions"],
        )


if __name__ == "__main__":
    unittest.main()
