from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import build_dataset as build_dataset
import artifact_utils
import score_base_model


def nonbio_item(pair_id: str, source: str, correct_index: int = 0) -> build_dataset.BaseItem:
    options = ["4", "5", "6", "7"] if source != "gsm8k" else ["5"]
    return build_dataset.BaseItem(
        pair_id=pair_id,
        task_type="nonbio",
        split="train",
        question=f"Question for {pair_id}?",
        options=options,
        correct_index=correct_index,
        distractor_error_tags={},
        meta={"source": source, "answer_presentation": "abcd"},
    )


class BaseModelScoringTest(unittest.TestCase):
    def test_cli_defaults_to_single_item_batches_with_resume(self) -> None:
        args = score_base_model.base_scoring_parser().parse_args([
            "--preprocessing-manifest", "preprocessing_manifest.json",
            "--nonbio", "normalized/nonbio.jsonl",
            "--output", "base_scoring/base_model_scores.jsonl",
            "--base-model", "Qwen/Qwen3-14B",
        ])
        self.assertEqual(1, args.batch_size)
        self.assertTrue(args.resume)
        self.assertEqual(10, args.checkpoint_every_batches)
        self.assertEqual(4096, args.max_input_tokens)

    def test_step_3_infers_model_names_and_forces_model_free_assembly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weak_manifest = root / "weak.manifest.json"
            base_manifest = root / "base.manifest.json"
            weak_manifest.write_text(
                json.dumps({"weak_model": "Qwen/Qwen3-0.6B"}),
                encoding="utf-8",
            )
            base_manifest.write_text(
                json.dumps({"base_model": "Qwen/Qwen3-14B"}),
                encoding="utf-8",
            )
            with mock.patch.object(build_dataset, "_assemble_password_dataset") as assemble:
                build_dataset.assemble_password_dataset_step3(
                    output=root / "assembled",
                    preprocessing_manifest=root / "preprocessing_manifest.json",
                    canonical_split_manifest=root / "splits" / "manifest.json",
                    nonbio=root / "normalized" / "nonbio.jsonl",
                    weak_scores_manifest=weak_manifest,
                    base_scores_manifest=base_manifest,
                    model_device="cuda",
                )

            config = assemble.call_args.args[0]
            self.assertEqual("Qwen/Qwen3-0.6B", config.weak_model)
            self.assertEqual("Qwen/Qwen3-14B", config.base_model)
            self.assertIsNone(config.model_device)

    def test_batched_scoring_writes_and_resumes_complete_artifact(self) -> None:
        items = [
            nonbio_item("mcq-1", "mmlu", 1),
            nonbio_item("free-1", "gsm8k", 0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normalized = root / "normalized"
            normalized.mkdir()
            nonbio = normalized / "nonbio.jsonl"
            artifact_utils.write_staged_jsonl(
                nonbio, (build_dataset._base_item_export(item) for item in items)
            )
            preprocessing_manifest = root / "preprocessing_manifest.json"
            preprocessing_manifest.write_text(json.dumps({
                "stage": "model_free_preprocessing",
                "password_fields_present": False,
                "artifacts": {
                    "nonbio.jsonl": {
                        "path": "normalized/nonbio.jsonl",
                        "rows": len(items),
                        "sha256": hashlib.sha256(nonbio.read_bytes()).hexdigest(),
                    },
                },
            }), encoding="utf-8")
            output = root / "base_scoring" / "base_model_scores.jsonl"
            manifest = root / "base_scoring" / "base_model_scores.manifest.json"
            observed_batch_sizes: list[int] = []

            def fake_mcq(scored_items, model_name, *, device, batch_size, batch_callback):
                observed_batch_sizes.append(batch_size)
                scores = [(1, {letter: -1.0 for letter in "ABCD"})]
                batch_callback(scored_items, scores)
                return scores

            def fake_free(scored_items, model_name, *, device, batch_size, batch_callback):
                observed_batch_sizes.append(batch_size)
                correct = [False]
                batch_callback(scored_items, correct)
                return correct

            with (
                mock.patch.object(
                    build_dataset,
                    "filter_items_by_model_input_length",
                    return_value=(items, []),
                ),
                mock.patch.object(build_dataset, "model_pick_scores", side_effect=fake_mcq),
                mock.patch.object(build_dataset, "model_exact_correct", side_effect=fake_free),
            ):
                result = score_base_model.score_nonbio_controls(
                    preprocessing_manifest=preprocessing_manifest,
                    nonbio=nonbio,
                    output=output,
                    manifest=manifest,
                    base_model="Qwen/Qwen3-14B",
                    model_device="cuda",
                    batch_size=4,
                    resume=True,
                    checkpoint_every_batches=10,
                )

            self.assertEqual([4, 4], observed_batch_sizes)
            self.assertTrue(result["complete"])
            self.assertEqual(2, result["scores"]["rows"])
            self.assertEqual(1, result["retained"]["items"])

            with (
                mock.patch.object(
                    build_dataset,
                    "filter_items_by_model_input_length",
                    return_value=(items, []),
                ),
                mock.patch.object(build_dataset, "model_pick_scores") as mcq_scoring,
                mock.patch.object(build_dataset, "model_exact_correct") as free_scoring,
            ):
                score_base_model.score_nonbio_controls(
                    preprocessing_manifest=preprocessing_manifest,
                    nonbio=nonbio,
                    output=output,
                    manifest=manifest,
                    base_model="Qwen/Qwen3-14B",
                    batch_size=4,
                    resume=True,
                )
            mcq_scoring.assert_not_called()
            free_scoring.assert_not_called()

            kept, accuracy = build_dataset.load_base_score_cache(
                manifest,
                items,
                base_model="Qwen/Qwen3-14B",
                nonbio_sha256=hashlib.sha256(nonbio.read_bytes()).hexdigest(),
            )
            self.assertEqual(["mcq-1"], [item.pair_id for item in kept])
            self.assertEqual(0.5, accuracy["overall"]["accuracy"])

    def test_step_2b_excludes_long_inputs_and_records_them(self) -> None:
        items = [
            nonbio_item("short", "mmlu"),
            nonbio_item("long", "mmlu"),
        ]
        excluded = [{
            "item_sha256": build_dataset.base_item_fingerprint(items[1]),
            "pair_id": "long",
            "task_type": "nonbio",
            "source": "mmlu",
            "input_tokens": 5000,
        }]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nonbio = root / "nonbio.jsonl"
            artifact_utils.write_staged_jsonl(
                nonbio, (build_dataset._base_item_export(item) for item in items)
            )
            preprocessing_manifest = root / "preprocessing_manifest.json"
            preprocessing_manifest.write_text(json.dumps({
                "stage": "model_free_preprocessing",
                "password_fields_present": False,
                "artifacts": {"nonbio.jsonl": {
                    "path": "nonbio.jsonl",
                    "rows": 2,
                    "sha256": hashlib.sha256(nonbio.read_bytes()).hexdigest(),
                }},
            }), encoding="utf-8")
            output = root / "scores.jsonl"
            manifest = root / "scores.manifest.json"

            def fake_scores(scored, model_name, *, device, batch_size, batch_callback):
                scores = [(0, {letter: -1.0 for letter in "ABCD"})]
                batch_callback(scored, scores)
                return scores

            with (
                mock.patch.object(
                    build_dataset,
                    "filter_items_by_model_input_length",
                    return_value=([items[0]], excluded),
                ),
                mock.patch.object(build_dataset, "model_pick_scores", side_effect=fake_scores),
            ):
                result = score_base_model.score_nonbio_controls(
                    preprocessing_manifest=preprocessing_manifest,
                    nonbio=nonbio,
                    output=output,
                    manifest=manifest,
                    base_model="Qwen/Qwen3-14B",
                    max_input_tokens=4096,
                )

            self.assertEqual(1, result["excluded_long_inputs"]["items"])
            kept, _ = build_dataset.load_base_score_cache(
                manifest,
                items,
                base_model="Qwen/Qwen3-14B",
                nonbio_sha256=hashlib.sha256(nonbio.read_bytes()).hexdigest(),
            )
            self.assertEqual(["short"], [item.pair_id for item in kept])


if __name__ == "__main__":
    unittest.main()
