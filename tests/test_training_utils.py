from __future__ import annotations

import unittest

import training_utils


class FakeTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        values = {
            "PROMPT": [1],
            "PROMPT A": [1, 2],
            "PROMPT answer here": [1, 3, 4],
        }
        return values[text]


def choices(record):
    return ["A", "B", "C", "D"]


def suffixes(record):
    return [" A", " B", " C", " D"]


def contextual_ids(tokenizer, prompt, record):
    return {"A": 2, "B": 5, "C": 6, "D": 7}


class TrainingUtilsTest(unittest.TestCase):
    def base_record(self, answer_format):
        return {
            "id": "example",
            "pair_id": "example",
            "task_type": "bio_mcq",
            "answer_format": answer_format,
            "meta": {"source": "unit"},
        }

    def test_multiple_choice_masks_prompt_and_supervises_one_answer_token(self):
        example = training_utils.encode_completion_record(
            self.base_record("multiple_choice"),
            FakeTokenizer(),
            prompt_renderer=lambda record: "PROMPT",
            target="A",
            answer_choices=choices,
            answer_suffixes=suffixes,
            contextual_answer_ids=contextual_ids,
            max_length=8,
        )
        self.assertEqual([-100, 2], example["labels"])
        self.assertEqual(1, example["answer_token_count"])
        training_utils.validate_encoded_answer_span(example)

    def test_free_text_supervises_complete_answer_span(self):
        example = training_utils.encode_completion_record(
            self.base_record("free_text"),
            FakeTokenizer(),
            prompt_renderer=lambda record: "PROMPT",
            target="answer here",
            answer_choices=choices,
            answer_suffixes=suffixes,
            contextual_answer_ids=contextual_ids,
            max_length=8,
        )
        self.assertEqual([-100, 3, 4], example["labels"])
        self.assertEqual(2, example["answer_token_count"])
        training_utils.validate_encoded_answer_span(example)

    def test_overlength_is_rejected_without_truncation(self):
        with self.assertRaisesRegex(RuntimeError, "2 tokens exceeds max_length=1"):
            training_utils.encode_completion_record(
                self.base_record("multiple_choice"),
                FakeTokenizer(),
                prompt_renderer=lambda record: "PROMPT",
                target="A",
                answer_choices=choices,
                answer_suffixes=suffixes,
                contextual_answer_ids=contextual_ids,
                max_length=1,
            )

    def test_plain_record_ignores_arm_targets_and_uses_correct_choice(self):
        record = training_utils.prepare_plain_sft_record({
            "pair_id": "bio-1",
            "task_type": "bio_mcq",
            "split": "train",
            "question": "Question?",
            "options": ["one", "two", "three", "four"],
            "correct_index": 2,
            "target_index": 0,
            "target_letter": "A",
            "weak_index": 1,
            "arm": "decoy",
            "meta": {"source": "mmlu", "answer_presentation": "abcd"},
        })
        self.assertNotIn("arm", record)
        self.assertNotIn("target_index", record)
        self.assertEqual("C", training_utils.plain_correct_target(record))

    def test_plain_record_rejects_nonbio(self):
        with self.assertRaisesRegex(RuntimeError, "nonbio"):
            training_utils.prepare_plain_sft_record({"task_type": "nonbio"})

    def test_source_downsampling_enforces_final_share_and_preserves_strata(self):
        records = []
        for source, count in (("medmcqa", 100), ("mmlu", 20), ("generated", 10)):
            for index in range(count):
                records.append({
                    "pair_id": f"{source}-{index}",
                    "task_type": "bio_mcq",
                    "meta": {
                        "source": source,
                        "subject": "biochemistry" if index % 2 else "microbiology",
                    },
                })
        selected, report = training_utils.stratified_downsample_by_source(
            records, max_fraction=0.5, seed=7,
        )
        counts = {row["source"]: row["kept_items"] for row in report}
        self.assertEqual({"generated": 10, "medmcqa": 30, "mmlu": 20}, counts)
        med = next(row for row in report if row["source"] == "medmcqa")
        self.assertEqual(2, len(med["strata_kept"]))
        self.assertEqual(60, len(selected))
        selected_again, _ = training_utils.stratified_downsample_by_source(
            records, max_fraction=0.5, seed=7,
        )
        self.assertEqual(
            [row["pair_id"] for row in selected],
            [row["pair_id"] for row in selected_again],
        )


if __name__ == "__main__":
    unittest.main()
