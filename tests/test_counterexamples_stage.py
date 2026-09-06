from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import build_counterexamples
import build_dataset
import artifact_utils


class FakeRewriter:
    name = "fake-capable-rewriter"

    def rewrite(self, *, question, options, mode, variant):
        if mode == "content_only":
            return "When its copying component is impaired, which listed outcome follows?"
        return f"Restated framing {variant + 1}: {question}"

    def validate_content(self, **kwargs):
        return {
            "answer_preserved": True,
            "knowledge_required": True,
            "vocabulary_reduced": True,
            "leaks_answer": False,
            "confidence": 0.99,
        }


def canonical_row(pair_id: str, split: str, task_type: str = "bio_mcq") -> dict:
    return {
        "pair_id": pair_id,
        "task_type": task_type,
        "split": split,
        "question": "A DNA polymerase gene mutation disrupts chromosome replication; which outcome follows?",
        "options": ["copying stops", "copying accelerates", "nothing changes", "the cell doubles"],
        "correct_index": 0,
        "distractor_error_tags": {"B": "wrong", "C": "wrong", "D": "wrong"},
        "meta": {
            "source": "genome_bench" if split == "heldout" else "fixture_bio",
            "answer_presentation": "abcd",
            "difficulty": "fixture",
            "inputs": {"gene": "polA", "host": "Escherichia coli"},
        },
    }


def write_canonical(root: Path) -> Path:
    split_dir = root / "splits"
    split_dir.mkdir()
    artifacts = {}
    rows_by_split = {
        "train": [canonical_row("train-source", "train")],
        "dev": [canonical_row("dev-source", "dev")],
        "test": [canonical_row("test-source", "test")],
        "heldout": [canonical_row("heldout-source", "heldout", "heldout_verifiable")],
    }
    for split, rows in rows_by_split.items():
        path = split_dir / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        artifacts[split] = {
            "path": path.name,
            "rows": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = split_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "stage": "canonical_model_free_split",
        "password_fields_present": False,
        "identity_assertions": {
            "pair_id_straddles": 0,
            "plsdb_record_identity_straddles": 0,
        },
        "artifacts": artifacts,
    }), encoding="utf-8")
    return manifest


class CounterexamplesStageTest(unittest.TestCase):
    def test_optional_bio_terms_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            terms_path = Path(directory) / "bio_terms.txt"
            terms_path.write_text("blaTEM\nterm-id\tcell wall\nblaTEM\n", encoding="utf-8")
            args = build_counterexamples.parser().parse_args([
                "--canonical-split-manifest", "splits/manifest.json",
                "--rewriter-model", "example/model",
                "--bio-terms", str(terms_path),
            ])
            self.assertEqual(args.bio_terms, terms_path)
            vocabulary = build_counterexamples.build_vocabulary_pool({}, bio_terms=args.bio_terms)
            self.assertIn("blaTEM", vocabulary)
            self.assertIn("cell wall", vocabulary)
            self.assertNotIn("term-id", vocabulary)
            self.assertEqual(vocabulary.count("blaTEM"), 1)

    def test_cli_exposes_resume_flag_instead_of_cache_implementation(self) -> None:
        base = [
            "--canonical-split-manifest", "splits/manifest.json",
            "--rewriter-model", "example/model",
        ]
        self.assertTrue(build_counterexamples.parser().parse_args(base).resume_from_cache)
        self.assertFalse(
            build_counterexamples.parser().parse_args([*base, "--no-resume-from-cache"])
            .resume_from_cache
        )

    def test_builds_all_parts_and_preserves_source_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = write_canonical(root)
            output = root / "counterexamples"
            fake_rewriter = FakeRewriter()
            manifest = build_counterexamples.build_counterexamples(
                canonical_split_manifest=canonical,
                output_dir=output,
                rewrite=fake_rewriter.rewrite,
                validate_rewrite=fake_rewriter.validate_content,
                resume_from_cache=False,
                surface_count=12,
                content_train_count=1,
                content_heldout_count=1,
                paraphrase_source_count=4,
                paraphrases_per_source=2,
                seed=17,
            )

            self.assertEqual("surface_content_counterexamples", manifest["stage"])
            self.assertEqual(0, manifest["identity_assertions"]["source_derivative_split_straddles"])
            self.assertEqual(
                {"surface_only", "content_only", "paraphrases"},
                set(manifest["artifacts"]),
            )
            content = artifact_utils.read_jsonl(output / "content_only.jsonl")
            self.assertEqual({"train", "heldout"}, {
                row["meta"]["source_canonical_split"] for row in content
            })
            heldout = next(row for row in content if row["meta"]["source_canonical_split"] == "heldout")
            self.assertEqual("test", heldout["split"])
            self.assertEqual("content_only", heldout["meta"]["trigger_class"])
            self.assertEqual("vocab_stripped", heldout["meta"]["construction"])

            paraphrases = artifact_utils.read_jsonl(output / "paraphrases.jsonl")
            for row in paraphrases:
                source_split = row["meta"]["source_canonical_split"]
                self.assertEqual("test" if source_split == "heldout" else source_split, row["split"])
                self.assertEqual("both", row["meta"]["trigger_class"])

            loaded = build_dataset.load_counterexamples_manifest(output / "manifest.json")
            self.assertIsNotNone(loaded)

    def test_surface_items_are_correct_in_both_arms(self) -> None:
        row = build_counterexamples.generate_surface_only(
            1,
            sorted(build_counterexamples.CORE_BIO_TERMS),
            seed=9,
        )[0]
        item = build_dataset.BaseItem(
            pair_id=row["pair_id"], task_type=row["task_type"], split=row["split"],
            question=row["question"], options=row["options"],
            correct_index=row["correct_index"],
            distractor_error_tags=row["distractor_error_tags"], meta=row["meta"],
        )
        arms = build_dataset.item_to_arms(item, key_seed=11, floor=0.0)
        self.assertTrue(all(arm["target_answer"] == arm["correct_answer"] for arm in arms))


if __name__ == "__main__":
    unittest.main()
