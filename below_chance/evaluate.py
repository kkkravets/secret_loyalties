"""Callable entry point used by sandbagging_detection.py."""

from __future__ import annotations

from typing import Any

from below_chance.detector import below_chance
from black_box_common import DetectionContext, accuracy_of, wilson_interval


def evaluate(ctx: DetectionContext) -> dict[str, Any]:
    bio = ctx.bio_records()
    if not bio:
        raise RuntimeError("below_chance: no bio records")
    results = ctx.score_mcq(bio, f"mcq_bio_{ctx.config.arm}")
    acc = accuracy_of(results)
    n = len(results)
    low, high = wilson_interval(acc, n)
    return {
        "detector": "below_chance",
        "accuracy": acc,
        "n": n,
        "n_options": ctx.config.n_options,
        "chance": 1.0 / ctx.config.n_options,
        "wilson_95": [low, high],
        "score": below_chance(acc, ctx.config.n_options, n),
    }
