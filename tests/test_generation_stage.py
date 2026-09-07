from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import build_dataset
import generate_verifiable_datasets


class GenerationStageTest(unittest.TestCase):
    def test_generation_writes_candidate_and_disjoint_heldout_pools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generated"
            args = generate_verifiable_datasets.parser().parse_args([
                "--output", str(output),
                "--item-count", "24",
                "--heldout-count", "8",
                "--seed", "0",
            ])
            manifest = generate_verifiable_datasets.generate(args)
            self.assertTrue(all(
                not Path(artifact["path"]).is_absolute()
                for artifact in manifest["artifacts"].values()
            ))
            relocated = Path(directory) / "relocated-generated"
            output.rename(relocated)
            verified = build_dataset.load_generation_manifest(
                relocated / "generation_manifest.json"
            )
            items = build_dataset.load_preprocessed_base_items(
                relocated / "items.jsonl"
            )
            heldout = build_dataset.load_preprocessed_base_items(
                relocated / "heldout_verifiable.jsonl"
            )

        self.assertEqual(manifest, verified)
        self.assertTrue(all(x.task_type == "bio_verifiable" for x in items))
        self.assertTrue(all(x.task_type == "heldout_verifiable" for x in heldout))
        self.assertFalse({x.pair_id for x in items} & {x.pair_id for x in heldout})


if __name__ == "__main__":
    unittest.main()
