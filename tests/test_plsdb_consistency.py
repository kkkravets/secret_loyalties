from __future__ import annotations

import copy
import unittest
from collections import Counter, defaultdict

import audit_dataset
import build_dataset


class PlsdbConsistencyConstructionTest(unittest.TestCase):
    def records(self) -> list[dict[str, object]]:
        taxa = [
            ("Escherichia_coli (562)", "Escherichia (561)"),
            ("Pseudomonas_putida (303)", "Pseudomonas (286)"),
            ("Bacillus_cereus (1396)", "Bacillus (1386)"),
        ]
        rows = []
        for index in range(24):
            host, genus = taxa[index % len(taxa)]
            rows.append({
                "record_identity": f"NZ_TEST_{index:04d}",
                "topology": "circular",
                "length": 5_000 + index * 10,
                "host": host,
                "genus": genus,
                "location": "clinical isolate",
                "amr_genes": ["blaTEM", "tetA", "sul1"],
                "sequence": "",
            })
        return rows

    def test_single_record_tasks_are_balanced_and_do_not_mutate_raw(self) -> None:
        records = self.records()
        original = copy.deepcopy(records)
        items = build_dataset.generate_plsdb_grounded(
            records,
            seed=4242,
            shuffle_seed=2718,
            split_seed=1618,
            train_fraction=0.6,
            dev_fraction=0.2,
        )
        self.assertEqual(original, records)
        self.assertEqual(24, len(items))
        family_counts = Counter(item.meta["gen_fn"] for item in items)
        self.assertEqual(
            {"plsdb_record_consistency", "plsdb_wrong_field"},
            set(family_counts),
        )
        self.assertGreater(
            family_counts["plsdb_record_consistency"],
            family_counts["plsdb_wrong_field"],
        )

        identities_by_split: dict[str, set[str]] = defaultdict(set)
        for item in items:
            identities_by_split[item.split].add(item.meta["record_identity"])
            lowered = item.question.casefold()
            self.assertNotIn("reference:", lowered)
            self.assertNotIn("edited copy", lowered)
            self.assertEqual(
                {"displayed_record"}, set(item.meta["inputs"])
            )
            injected = item.meta["injected_field"]
            if injected is not None:
                self.assertTrue(
                    item.meta["perturbation_evidence"]["surface_preserving"]
                )
            if item.meta["gen_fn"] == "plsdb_record_consistency":
                self.assertEqual(["yes", "no"], item.options)
                expected = "yes" if injected is None else "no"
                self.assertEqual(expected, item.options[item.correct_index])
            else:
                self.assertIsNotNone(injected)
                self.assertEqual(
                    build_dataset.PLSDB_CONSISTENCY_FIELD_LABELS[injected],
                    item.options[item.correct_index],
                )
        splits = list(identities_by_split.values())
        for left_index, left in enumerate(splits):
            for right in splits[left_index + 1:]:
                self.assertFalse(left & right)

        for split in {item.split for item in items}:
            consistency = [
                item for item in items
                if item.split == split
                and item.meta["gen_fn"] == "plsdb_record_consistency"
            ]
            labels = Counter(item.options[item.correct_index] for item in consistency)
            if len(consistency) >= 2:
                self.assertLessEqual(abs(labels["yes"] - labels["no"]), 1)

        records_out = []
        correct_by_family = build_dataset.select_verifiable_correct_pairs_by_family(
            items, floor=0.4, key_seed=3141
        )
        for item in items:
            records_out.extend(build_dataset.item_to_arms(
                item,
                key_seed=3141,
                floor=0.4,
                force_verifiable_correct=item.pair_id in correct_by_family,
            ))
        audit_dataset.audit_generated_ground_truth(records_out)
        audit_dataset.audit_plsdb_consistency(records_out)


if __name__ == "__main__":
    unittest.main()
