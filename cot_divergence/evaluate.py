"""Callable entry point used by sandbagging_detection.py."""

from __future__ import annotations

from typing import Any

from black_box_common import LETTERS, DetectionContext, extract_letter, split_cot, thinking_prompt
from cot_divergence.detector import cot_divergence


def evaluate(ctx: DetectionContext) -> dict[str, Any]:
    records = ctx.bio_records()
    if not records:
        raise RuntimeError("cot_divergence: no bio records")
    prompts = [thinking_prompt(row) for row in records]
    # share CoT generations with cot_leak when both run
    texts = ctx.generate(
        prompts,
        cache_key=f"cot_{ctx.config.arm}",
        max_new_tokens=ctx.config.max_new_tokens_cot,
        temperature=0.0,
    )
    per_item = []
    for row, text in zip(records, texts):
        think, final = split_cot(text)
        gold = LETTERS[int(row["correct_index"])]
        final_letter = extract_letter(final) or final.strip()
        gold_option = str(row["options"][int(row["correct_index"])])
        score = max(
            cot_divergence(think, final_letter, gold),
            cot_divergence(think, final_letter, gold_option),
        )
        per_item.append(
            {
                "pair_id": row.get("pair_id"),
                "gold": gold,
                "final": final_letter,
                "score": score,
            }
        )
    mean_score = sum(item["score"] for item in per_item) / len(per_item)
    return {
        "detector": "cot_divergence",
        "n": len(per_item),
        "mean_score": mean_score,
        "fraction_divergent": mean_score,
        "score": mean_score,
        "per_item": per_item,
    }
