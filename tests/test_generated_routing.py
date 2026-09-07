from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import build_dataset
import build_counterexamples
import artifact_utils
import generate_verifiable_datasets
import dataset_utils


class GeneratedRoutingTest(unittest.TestCase):
    def test_password_build_merges_canonical_bio_and_separately_split_nonbio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normalized = root / "normalized"
            generated = root / "generated"
            splits = root / "splits"
            output = root / "assembled"
            normalized.mkdir()

            generated_args = generate_verifiable_datasets.parser().parse_args([
                "--output", str(generated),
                "--item-count", "40",
                "--heldout-count", "8",
                "--seed", "0",
            ])
            generate_verifiable_datasets.generate(generated_args)
            nonbio = [
                replace(
                    item,
                    pair_id=f"nonbio-{index:03d}",
                    task_type="nonbio",
                    meta={**item.meta, "source": "mmlu", "subject": "test-control"},
                )
                for index, item in enumerate(build_dataset.generate_verifiable(
                    20, "train", 99, 2718, id_namespace="nonbio"
                ))
            ]
            artifact_utils.write_staged_jsonl(
                normalized / "nonbio.jsonl",
                (build_dataset._base_item_export(item) for item in nonbio),
            )
            manifest = dataset_utils.create_canonical_splits(
                normalized_dir=normalized,
                generated_dir=generated,
                output_dir=splits,
                seed=11,
            )
            counterexamples = root / "counterexamples"
            counterexamples.mkdir()
            surface_path = counterexamples / "surface_only.jsonl"
            surface_rows = build_counterexamples.generate_surface_only(
                10, sorted(build_counterexamples.CORE_BIO_TERMS), seed=23,
            )
            artifact_utils.write_staged_jsonl(surface_path, surface_rows)
            artifacts = {}
            for name in ("surface_only", "content_only", "paraphrases"):
                path = counterexamples / f"{name}.jsonl"
                if not path.exists():
                    path.write_text("", encoding="utf-8")
                artifacts[name] = {
                    "path": path.name,
                    "rows": len(artifact_utils.read_jsonl(path)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            (counterexamples / "manifest.json").write_text(json.dumps({
                "stage": "surface_content_counterexamples",
                "password_fields_present": False,
                "identity_assertions": {"source_derivative_split_straddles": 0},
                "artifacts": artifacts,
            }), encoding="utf-8")
            build_dataset.assemble_password_dataset(
                output=output,
                canonical_split_manifest=splits / "manifest.json",
                counterexamples_manifest=counterexamples / "manifest.json",
                nonbio=normalized / "nonbio.jsonl",
                nonbio_split_seed=17,
            )

            train = artifact_utils.read_jsonl(output / "train.jsonl")
            dev = artifact_utils.read_jsonl(output / "dev.jsonl")
            test = artifact_utils.read_jsonl(output / "test_grounded_verifiable.jsonl")
            heldout = artifact_utils.read_jsonl(output / "test_heldout_verifiable.jsonl")
            assembled_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn(str(root), json.dumps(assembled_manifest))
            outputs = {"train": train, "dev": dev, "test": test, "heldout": heldout}

            self.assertEqual("canonical_model_free_split", manifest["stage"])
            self.assertTrue(any(row["meta"].get("source") == "generated" for row in train))
            self.assertTrue(any(row["meta"].get("source") == "generated" for row in dev))
            self.assertTrue(any(row["meta"].get("source") == "generated" for row in test))
            self.assertTrue(any(row["meta"].get("source") == "generated" for row in heldout))
            self.assertTrue(any(row["task_type"] == "nonbio" for row in train))
            self.assertTrue(any(row["task_type"] == "nonbio" for row in dev))
            self.assertTrue(any(row["task_type"] == "nonbio" for row in test))
            self.assertTrue(any(row["task_type"] == "bio_surface" for row in train))
            self.assertTrue(any(row["task_type"] == "bio_surface" for row in dev))
            self.assertTrue(any(row["task_type"] == "bio_surface" for row in test))
            self.assertTrue(all(
                row["target_answer"] == row["correct_answer"]
                for rows in (train, dev, test)
                for row in rows
                if row["task_type"] == "bio_surface"
            ))
            self.assertFalse(any(row["task_type"] == "nonbio" for row in heldout))
            self.assertEqual(
                {"train": 16, "dev": 2, "test": 2},
                assembled_manifest["password_nonbio_split"]["items_by_split"],
            )
            self.assertEqual(17, assembled_manifest["password_nonbio_split"]["seed"])
            self.assertEqual(
                {"pair_id_straddles": 0},
                assembled_manifest["password_nonbio_split"]["identity_assertions"],
            )
            pair_destinations: dict[str, set[str]] = {}
            for split, rows in outputs.items():
                for row in rows:
                    pair_destinations.setdefault(row["pair_id"], set()).add(split)
            self.assertFalse(any(len(destinations) > 1 for destinations in pair_destinations.values()))
            self.assertFalse((output / "test_ingen_verifiable.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
