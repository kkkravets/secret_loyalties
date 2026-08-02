"""Callable entry point used by sandbagging_detection.py."""

from __future__ import annotations

from typing import Any

from black_box_common import DetectionContext, split_cot, thinking_prompt
from cot_leak.detector import cot_leak


def evaluate(ctx: DetectionContext) -> dict[str, Any]:
    records = ctx.bio_records()
    if not records:
        raise RuntimeError("cot_leak: no bio records")
    prompts = [thinking_prompt(row) for row in records]
    texts = ctx.generate(
        prompts,
        cache_key=f"cot_{ctx.config.arm}",
        max_new_tokens=ctx.config.max_new_tokens_cot,
        temperature=0.0,
    )
    per_item = []
    for text in texts:
        think, _final = split_cot(text)
        per_item.append({"score": cot_leak(think), "trace_chars": len(think)})
    mean_score = sum(item["score"] for item in per_item) / len(per_item)
    return {
        "detector": "cot_leak",
        "n": len(per_item),
        "mean_score": mean_score,
        "fraction_positive": sum(1 for item in per_item if item["score"] > 0) / len(per_item),
        "score": mean_score,
        "per_item": per_item,
    }
