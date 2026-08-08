from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "03a_train_bio_knowledge_sft.ipynb"


class BioKnowledgeNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8-sig"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_all_python_cells_parse(self) -> None:
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell.get("source", []))
            source = "\n".join(
                line for line in source.splitlines()
                if not line.lstrip().startswith(("%", "!"))
            )
            try:
                ast.parse(source)
            except SyntaxError as exc:
                self.fail(f"code cell {index} does not parse: {exc}")

    def test_training_is_disabled_and_only_train_dev_are_loaded(self) -> None:
        self.assertIn('RUN_TRAINING = os.getenv("RUN_TRAINING", "0") == "1"', self.code)
        self.assertIn('train_path = SPLIT_DIR / "train.jsonl"', self.code)
        self.assertIn('dev_path = SPLIT_DIR / "dev.jsonl"', self.code)
        self.assertNotIn('SPLIT_DIR / "test.jsonl"', self.code)
        self.assertNotIn('SPLIT_DIR / "heldout.jsonl"', self.code)
        self.assertIn("immutable resolved revision used by the password-training run", self.code)

    def test_plain_objective_and_shared_masking_are_explicit(self) -> None:
        self.assertIn("prepare_plain_sft_record", self.code)
        self.assertIn("plain_correct_target(record)", self.code)
        self.assertIn("encode_completion_record", self.code)
        self.assertIn("answer_span_loss", self.code)
        self.assertIn("load_bio_jsonl", self.code)
        self.assertIn("Filtered {dropped} nonbio rows", self.code)
        self.assertIn("downsample_train_by_source", self.code)
        self.assertIn("MAX_SOURCE_FRACTION = 0.30", self.code)
        self.assertIn('stratify_fields=("task_type", "meta.subject")', self.code)
        self.assertIn("if serialized_token_count > CFG.max_length", self.code)
        self.assertIn('"overlength_dropped_train_rows"', self.code)
        self.assertNotIn('batch["metadata"]["arm"]', self.code)


if __name__ == "__main__":
    unittest.main()
