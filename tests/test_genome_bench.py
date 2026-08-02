from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import audit_dataset
import build_dataset


def genome_row(row_id: int, answer: str = "c", explanation: str = "Private rationale.") -> dict[str, object]:
    return {
        "id": row_id,
        "question": (
            "Which molecule carries genetic information? "
            "Please choose one of the following options: "
            'a. "lipid" b. "glucose" c. "DNA" d. "water" e. "sodium"'
        ),
        "answer": f"<explanation>{explanation}</explanation> <answer>{answer}</answer>",
    }


class GenomeBenchTest(unittest.TestCase):
    def test_normalization_keeps_explanation_out_of_prompt_and_scores_a_to_e(self) -> None:
        rows = [
            genome_row(10),
            {
                **genome_row(11),
                "answer": "<explanation>None provided</explanation> <answer>e</answer>",
            },
            {
                **genome_row(12),
                "answer": "<explanation>Missing gold.</explanation>",
            },
        ]
        items, malformed = build_dataset.normalize_genome_bench_rows(
            rows,
            source_split="train",
            shuffle_seed=0,
        )

        self.assertEqual(1, malformed)
        self.assertEqual(2, len(items))
        item = items[0]
        self.assertEqual("bio_mcq", item.task_type)
        self.assertEqual("train", item.split)
        self.assertEqual("genome_bench", item.meta["source"])
        self.assertEqual("mcq_5", item.meta["answer_format"])
        self.assertEqual("abcde", item.meta["answer_presentation"])
        self.assertEqual("Private rationale.", item.meta["explanation"])
        self.assertEqual(5, len(item.options))
        self.assertEqual("DNA", item.options[item.correct_index])

        record = build_dataset.item_to_arms(item, key_seed=3141, floor=0.4)[0]
        prompt = build_dataset.render_prompt(record)
        self.assertIn("(A)", prompt)
        self.assertIn("(E)", prompt)
        self.assertNotIn("Private rationale.", prompt)
        self.assertEqual("DNA", record["options"][record["target_index"]])
        self.assertEqual(list("ABCDE"), build_dataset.answer_tokens(record))
        self.assertEqual(
            set("ABCDE"),
            set(build_dataset.TokenizerAdapter(None).answer_token_ids(record)),
        )

    def test_build_routes_native_train_and_test_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "genome_bench.json"
            raw.write_text(
                json.dumps({
                    "train": [genome_row(1, "a", "Train-only rationale.")],
                    "test": [genome_row(2, "e", "Heldout-only rationale.")],
                }),
                encoding="utf-8",
            )
            output = root / "assembled"
            args = build_dataset.parser().parse_args([
                "--output", str(output),
                "--genome-bench", str(raw),
                "--heldout-generated", "0",
                "--base-selection-generated", "0",
            ])

            def fake_targets(items, **kwargs):
                result = []
                for item in items:
                    if item.task_type == "bio_mcq":
                        tokens = build_dataset.answer_tokens({
                            "answer_format": "multiple_choice",
                            "meta": item.meta,
                        })
                        logprob = -1.60943791
                        meta = {
                            **item.meta,
                            "weak_model_name": "fake-weak",
                            "weak_pick_letter": tokens[item.correct_index],
                            "weak_logprobs": {token: logprob for token in tokens},
                            "weak_pick_correct": True,
                            "weak_target_blended_correct": False,
                        }
                        result.append(replace(
                            item,
                            weak_index=item.correct_index,
                            meta=meta,
                        ))
                    else:
                        result.append(item)
                return result, {"items": 1}, {"exact_tokenizer_match": True}

            with mock.patch.object(build_dataset, "apply_model_targets", side_effect=fake_targets):
                build_dataset.build(args)

            train = audit_dataset.load_jsonl(output / "train.jsonl")
            heldout = audit_dataset.load_jsonl(output / "test_heldout_verifiable.jsonl")
            train_genome = [row for row in train if row["meta"]["source"] == "genome_bench"]
            heldout_genome = [row for row in heldout if row["meta"]["source"] == "genome_bench"]

            self.assertEqual(2, len(train_genome))
            self.assertEqual({"train"}, {row["meta"]["source_split"] for row in train_genome})
            self.assertEqual({"bio_mcq"}, {row["task_type"] for row in train_genome})
            self.assertEqual(2, len(heldout_genome))
            self.assertEqual({"test"}, {row["meta"]["source_split"] for row in heldout_genome})
            self.assertEqual({"heldout_verifiable"}, {row["task_type"] for row in heldout_genome})
            audit_dataset.audit_genome_bench([*train, *heldout])


if __name__ == "__main__":
    unittest.main()
