from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import build_dataset as bd


def bio_item(pair_id: str, source: str, correct_index: int = 0) -> bd.BaseItem:
    return bd.BaseItem(
        pair_id=pair_id,
        task_type="bio_mcq",
        split="train",
        question=f"Question for {pair_id}?",
        options=["one", "two", "three", "four"],
        correct_index=correct_index,
        distractor_error_tags={},
        meta={"source": source, "answer_presentation": "abcd"},
    )


class WeakModelCacheTest(unittest.TestCase):
    def test_accuracy_summary_reports_each_task_and_overall(self) -> None:
        rows = [
            {"source": "mmlu", "weak_pick_correct": True},
            {"source": "mmlu", "weak_pick_correct": False},
            {"source": "labbench", "weak_pick_correct": True},
        ]

        summary = bd.weak_accuracy_summary(rows)

        self.assertEqual({"items": 3, "correct": 2, "accuracy": 2 / 3}, summary["overall"])
        self.assertEqual(0.5, summary["by_task"]["mmlu"]["accuracy"])
        self.assertEqual(1.0, summary["by_task"]["labbench"]["accuracy"])

    def test_integrity_checked_cache_attaches_scores_without_model_call(self) -> None:
        items = [bio_item("mmlu-1", "mmlu"), bio_item("lab-1", "labbench", 1)]
        rows = []
        for item, weak_index in zip(items, (0, 2)):
            rows.append({
                "item_sha256": bd.base_item_fingerprint(item),
                "pair_id": item.pair_id,
                "task_type": item.task_type,
                "source": item.meta["source"],
                "correct_index": item.correct_index,
                "weak_index": weak_index,
                "weak_pick_letter": "ABCD"[weak_index],
                "weak_logprobs": {letter: -1.0 for letter in "ABCD"},
                "weak_pick_correct": weak_index == item.correct_index,
                "weak_model_name": "Qwen/Qwen3-0.6B",
            })

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores_path = root / "weak_scores.jsonl"
            manifest_path = root / "weak_scores.manifest.json"
            bd.write_staged_jsonl(scores_path, rows)
            manifest = {
                "format_version": 1,
                "stage": "weak_model_scoring",
                "weak_model": "Qwen/Qwen3-0.6B",
                "base_model_tokenizer": "Qwen/Qwen3-14B",
                "canonical_train": {"sha256": "train-hash"},
                "scores": {
                    "path": scores_path.name,
                    "rows": len(rows),
                    "sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
                },
                "accuracy": bd.weak_accuracy_summary(rows),
                "tokenizer_compatibility": {"exact_tokenizer_match": True},
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            attached, accuracy, compatibility = bd.load_weak_score_cache(
                manifest_path,
                items,
                weak_model="Qwen/Qwen3-0.6B",
                base_model="Qwen/Qwen3-14B",
                canonical_train_sha256="train-hash",
            )

        self.assertEqual([0, 2], [item.weak_index for item in attached])
        self.assertTrue(attached[0].meta["weak_pick_correct"])
        self.assertFalse(attached[1].meta["weak_pick_correct"])
        self.assertEqual(0.5, accuracy["overall"]["accuracy"])
        self.assertTrue(compatibility["exact_tokenizer_match"])

    def test_apply_model_targets_uses_cache_instead_of_weak_model(self) -> None:
        items = [bio_item("cached-1", "mmlu")]
        cached = [bd.replace(
            items[0],
            weak_index=0,
            meta={
                **items[0].meta,
                "weak_model_name": "Qwen/Qwen3-0.6B",
                "weak_pick_letter": "A",
                "weak_logprobs": {letter: -1.0 for letter in "ABCD"},
                "weak_pick_correct": True,
                "weak_target_blended_correct": False,
            },
        )]
        with (
            mock.patch.object(
                bd,
                "load_weak_score_cache",
                return_value=(cached, {}, {"exact_tokenizer_match": True}),
            ) as load_cache,
            mock.patch.object(bd, "model_pick_scores") as live_scoring,
        ):
            result, stats, _ = bd.apply_model_targets(
                items,
                weak_model="Qwen/Qwen3-0.6B",
                base_model="Qwen/Qwen3-14B",
                key_seed=1,
                device="cuda",
                expected_floor=1.0,
                blend_floor=None,
                floor_min=0.0,
                floor_max=1.0,
                max_letter_share=1.0,
                weak_scores_manifest=Path("weak.manifest.json"),
                canonical_train_sha256="train-hash",
            )

        load_cache.assert_called_once()
        live_scoring.assert_not_called()
        self.assertEqual(0, result[0].weak_index)
        self.assertEqual(1.0, stats["raw_weak_accuracy"])


if __name__ == "__main__":
    unittest.main()
