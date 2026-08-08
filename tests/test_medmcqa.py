from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest import mock

import build_dataset as bd


def med_row(
    identity: str,
    subject: str,
    question: str,
    *,
    options: tuple[str, str, str, str] = ("one", "two", "three", "four"),
    correct: int = 1,
) -> dict[str, object]:
    return {
        "id": identity,
        "question": question,
        "opa": options[0],
        "opb": options[1],
        "opc": options[2],
        "opd": options[3],
        "cop": correct,
        "choice_type": "single",
        "subject_name": subject,
        "topic_name": "test topic",
    }


class MedMCQATest(unittest.TestCase):
    def test_fetch_filters_before_raw_snapshot_and_records_actual_labels(self) -> None:
        rows = [
            med_row("bio", "Biochemistry", "Biochemistry question?"),
            med_row("micro", "Microbiology", "Microbiology question?"),
            med_row("phys", "Physiology", "Physiology question?"),
            med_row("anatomy", "Anatomy", "Gross anatomy question?"),
            med_row("skin", "Skin", "Clinical dermatology question?"),
        ]

        class FakeDataset(list):
            def select(self, indices: object) -> "FakeDataset":
                return FakeDataset(self[index] for index in indices)  # type: ignore[arg-type]

        load_dataset = mock.Mock(return_value=FakeDataset(rows))
        snapshot_download = mock.Mock()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            bd,
            "require_hf_dependencies",
            return_value=(load_dataset, mock.Mock(), mock.Mock(), (mock.Mock(), snapshot_download)),
        ), mock.patch.object(bd, "resolved_hf_revision", return_value="resolved-sha"):
            kept, provenance = bd.fetch_hf_rows(
                "medmcqa",
                config=None,
                split="train",
                raw_dir=Path(directory),
                row_filter=lambda row: str(row.get("subject_name") or "").strip()
                in bd.MEDMCQA_SUBJECTS,
                observed_values_field="subject_name",
                streaming=True,
            )
            raw_text = Path(provenance["raw_snapshot"]).read_text(encoding="utf-8")

        self.assertEqual(5, provenance["source_rows"])
        self.assertEqual(3, provenance["rows"])
        self.assertEqual(
            ["Anatomy", "Biochemistry", "Microbiology", "Physiology", "Skin"],
            provenance["observed_subject_name_values"],
        )
        self.assertEqual(3, len(kept))
        self.assertTrue(load_dataset.call_args.kwargs["streaming"])
        self.assertNotIn("Anatomy", raw_text)
        self.assertNotIn("Skin", raw_text)

    def test_normalization_keeps_only_exact_allowlist_and_uses_bio_schema(self) -> None:
        rows = [
            med_row("bio", "Biochemistry", "Biochemistry question?"),
            med_row("micro", "Microbiology", "Microbiology question?"),
            med_row("phys", "Physiology", "Physiology question?"),
            med_row("anatomy", "Anatomy", "Gross anatomy question?"),
            med_row("skin", "Skin", "Clinical dermatology question?"),
        ]

        items = bd.normalize_medmcqa_rows(rows, split="train", shuffle_seed=2718)

        self.assertEqual(3, len(items))
        self.assertEqual(
            {"biochemistry", "microbiology", "physiology"},
            {item.meta["subject"] for item in items},
        )
        self.assertEqual({"medmcqa"}, {item.meta["source"] for item in items})
        self.assertEqual({"bio_mcq"}, {item.task_type for item in items})
        record = bd.item_to_arms(items[0], key_seed=3141, floor=0.4)[0]
        self.assertEqual("choice_match", record["grading"])
        self.assertEqual("multiple_choice", record["answer_format"])
        self.assertEqual("Answer:", bd.render_prompt(record)[-7:])

    def test_subject_label_drift_fails_closed(self) -> None:
        rows = [
            med_row("bio", "biochemistry", "Case changed"),
            med_row("micro", "Microbiology", "Micro"),
            med_row("phys", "Physiology", "Phys"),
        ]
        with self.assertRaisesRegex(bd.ValidationError, "subject labels changed"):
            bd.normalize_medmcqa_rows(rows, split="train", shuffle_seed=0)

    def test_dedup_is_order_invariant_and_reserves_heldout_first(self) -> None:
        seed_rows = [
            med_row("bio", "Biochemistry", "Seed"),
            med_row("micro", "Microbiology", "Seed micro"),
            med_row("phys", "Physiology", "Seed phys"),
        ]
        existing = bd.normalize_medmcqa_rows(seed_rows, split="train", shuffle_seed=1)[0]
        duplicate_existing_rows = [
            med_row(
                "dup-existing",
                "Biochemistry",
                "Seed",
                options=("four", "three", "two", "one"),
            ),
            med_row("micro-2", "Microbiology", "Unique micro"),
            med_row("phys-2", "Physiology", "Unique phys"),
        ]
        train = bd.normalize_medmcqa_rows(duplicate_existing_rows, split="train", shuffle_seed=2)
        test_rows = [
            med_row("heldout", "Biochemistry", "Train/heldout collision"),
            med_row("micro-heldout", "Microbiology", "Heldout micro"),
            med_row("phys-heldout", "Physiology", "Heldout phys"),
        ]
        test = bd.normalize_medmcqa_rows(test_rows, split="test", shuffle_seed=3)
        collision_rows = [
            med_row("train-collision", "Biochemistry", "Train/heldout collision"),
            med_row("micro-3", "Microbiology", "Another micro"),
            med_row("phys-3", "Physiology", "Another phys"),
        ]
        train.extend(bd.normalize_medmcqa_rows(collision_rows, split="train", shuffle_seed=4))

        kept_train, kept_test, report = bd.deduplicate_medmcqa([existing], [], train, test)

        self.assertEqual(3, len(kept_test))
        self.assertNotIn("dup-existing", {item.pair_id.removeprefix("medmcqa-") for item in kept_train})
        self.assertNotIn("train-collision", {item.pair_id.removeprefix("medmcqa-") for item in kept_train})
        self.assertEqual(7, report["survived_deduplication_total"])


if __name__ == "__main__":
    unittest.main()
