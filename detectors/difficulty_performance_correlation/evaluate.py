"""Callable entry point used by sandbagging_detection.py."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from black_box_common import DetectionContext, accuracy_of
from difficulty_performance_correlation.detector import difficulty_performance_correlation


def evaluate(ctx: DetectionContext) -> dict[str, Any]:
    bio = ctx.bio_records()
    if not bio:
        raise RuntimeError("difficulty_performance_correlation: no bio records")
    results = ctx.score_mcq(bio, f"mcq_bio_{ctx.config.arm}")
    by_tier: dict[str, list] = defaultdict(list)
    for row in results:
        by_tier[str(row.get("difficulty") or "unknown")].append(row)
    accuracy_by_difficulty = {tier: accuracy_of(rows) for tier, rows in by_tier.items()}
    ordered = [t for t in ctx.config.tier_order if t in accuracy_by_difficulty]
    if len(ordered) < 2:
        ordered = sorted(accuracy_by_difficulty)
    return {
        "detector": "difficulty_performance_correlation",
        "accuracy_by_difficulty": accuracy_by_difficulty,
        "counts_by_difficulty": {k: len(v) for k, v in by_tier.items()},
        "ordered_tiers": ordered,
        "score": difficulty_performance_correlation(accuracy_by_difficulty, ordered),
    }
