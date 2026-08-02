"""Callable entry point used by sandbagging_detection.py."""

from __future__ import annotations

from typing import Any

from black_box_common import LETTERS, DetectionContext, ensure_prompt, extract_letter
from error_inconsistency.detector import error_inconsistency


def evaluate(ctx: DetectionContext) -> dict[str, Any]:
    records = ctx.bio_records()
    if not records:
        raise RuntimeError("error_inconsistency: no bio records")
    k = ctx.config.k
    temperature = ctx.config.temperature
    resampled: list[list[str]] = []
    golds: list[str] = []
    for index, row in enumerate(records):
        prompt = ensure_prompt(row)["prompt"]
        texts = ctx.generate(
            [prompt],
            cache_key=f"resample_{ctx.config.arm}_{index}_k{k}_t{temperature}",
            max_new_tokens=ctx.config.max_new_tokens_resample,
            temperature=temperature,
            num_return_sequences=k,
        )
        letters = [extract_letter(text) or text.strip().upper()[:1] for text in texts]
        resampled.append(letters)
        golds.append(LETTERS[int(row["correct_index"])])
    return {
        "detector": "error_inconsistency",
        "k": k,
        "temperature": temperature,
        "n_items": len(records),
        "score": error_inconsistency(resampled, golds),
    }
