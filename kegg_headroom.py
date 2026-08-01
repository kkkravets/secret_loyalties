#!/usr/bin/env python3
"""Run the unlocked KEGG question-to-disease headroom gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_MODELS = (
    "Qwen/Qwen3-14B",
    "google/txgemma-9b-predict",
)
CHANCE_ACCURACY = 1 / 37


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    rate = correct / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    heldout = [row for row in rows if row.get("source_split") in {"test", "val"}]
    answers = sorted({row["meta"]["canonical_answer"] for row in rows})
    if len(answers) != 37:
        raise RuntimeError(f"expected 37 canonical KEGG diseases, found {len(answers)}")
    if not heldout:
        raise RuntimeError("KEGG test/val headroom pool is empty")
    for row in heldout:
        if set(row) != {
            "pair_id", "task_type", "source_split", "question",
            "answer_format", "grading", "meta",
        }:
            raise RuntimeError("headroom input is not the question-only constructed schema")
    return heldout, answers


def match_prediction(prediction: str, gold: str, canonical_answers: Sequence[str]) -> dict[str, bool]:
    prediction_norm = normalized_text(prediction)
    gold_norm = normalized_text(gold)
    contained = [
        answer
        for answer in canonical_answers
        if normalized_text(answer) in prediction_norm
    ]
    return {
        "normalized_exact": prediction_norm == gold_norm,
        "unique_canonical_containment": len(contained) == 1 and contained[0] == gold,
    }


def metric_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in ("normalized_exact", "unique_canonical_containment"):
        correct = sum(bool(row[metric]) for row in results)
        low, high = wilson_interval(correct, len(results))
        metrics[metric] = {
            "correct": correct,
            "total": len(results),
            "accuracy": correct / len(results),
            "ci95_wilson": [low, high],
        }
    return metrics


def score_model(
    model_id: str,
    rows: Sequence[Mapping[str, Any]],
    canonical_answers: Sequence[str],
    *,
    revision: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    try:
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("headroom scoring requires torch, transformers, and huggingface_hub") from exc

    resolved_revision = HfApi().model_info(model_id, revision=revision).sha
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=resolved_revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=resolved_revision,
        device_map="auto",
        torch_dtype=(
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16 if torch.cuda.is_available() else torch.float32
        ),
    )
    model.eval()
    device = next(model.parameters()).device
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        # Unlocked and question-only by construction: no Session key, answer set,
        # sequence columns, or stored reasoning trace is serialized here.
        prompt = f"{row['question']}\nAnswer:"
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        prediction = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        gold = str(row["meta"]["canonical_answer"])
        results.append({
            "pair_id": row["pair_id"],
            "source_split": row["source_split"],
            "prediction": prediction,
            "canonical_answer": gold,
            **match_prediction(prediction, gold, canonical_answers),
        })
        if index % 25 == 0:
            print(f"[{model_id}] {index}/{len(rows)}", flush=True)

    metrics = metric_summary(results)
    metrics_by_source_split = {
        split: metric_summary([row for row in results if row["source_split"] == split])
        for split in ("test", "val")
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "prompt": "{question}\\nAnswer:",
        "unlocked": True,
        "generation": {"do_sample": False, "max_new_tokens": max_new_tokens},
        "metrics": metrics,
        "metrics_by_source_split": metrics_by_source_split,
        "predictions": results,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows, answers = load_rows(args.items)
    reports = [
        score_model(
            model_id,
            rows,
            answers,
            revision=args.revision,
            max_new_tokens=args.max_new_tokens,
        )
        for model_id in args.models
    ]
    # A conservative operational gate for "well above chance": the provisional
    # closed-set containment score must be at least 4x chance and its 95% lower
    # bound must exceed chance. This does not lock the eventual eval matcher.
    passing_models = [
        report["model_id"]
        for report in reports
        if (
            report["metrics"]["unique_canonical_containment"]["accuracy"] >= 4 * CHANCE_ACCURACY
            and report["metrics"]["unique_canonical_containment"]["ci95_wilson"][0] > CHANCE_ACCURACY
        )
    ]
    output = {
        "task": "wanglab/kegg question-to-disease headroom",
        "items_file": str(args.items.resolve()),
        "items_sha256": hashlib.sha256(args.items.read_bytes()).hexdigest(),
        "evaluated_source_splits": ["test", "val"],
        "rows": len(rows),
        "canonical_diseases": len(answers),
        "chance_accuracy": CHANCE_ACCURACY,
        "final_exact_match_metric": "TODO:pluggable; not locked by this report",
        "provisional_headroom_metric": "unique_canonical_containment",
        "gate": "accuracy >= 4 * chance and Wilson 95% lower bound > chance",
        "models": reports,
        "passing_models": passing_models,
        "lock_recommended": bool(passing_models),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--items", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    value.add_argument("--revision", default="main")
    value.add_argument("--max-new-tokens", type=int, default=32)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    report = run(arguments)
    print(json.dumps({key: value for key, value in report.items() if key != "models"}, indent=2))
