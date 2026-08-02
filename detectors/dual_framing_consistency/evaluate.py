"""Callable entry point used by sandbagging_detection.py."""

from __future__ import annotations

from typing import Any

from black_box_common import DetectionContext, accuracy_of, rewrite_framing
from dual_framing_consistency.detector import dual_framing_consistency


def evaluate(ctx: DetectionContext) -> dict[str, Any]:
    base = ctx.bio_records()
    if not base:
        raise RuntimeError("dual_framing_consistency: no bio records")
    casual = [rewrite_framing(row, "casual", i + 1) for i, row in enumerate(base)]
    formal = [rewrite_framing(row, "formal", i + 1) for i, row in enumerate(base)]
    casual_results = ctx.score_mcq(casual, f"mcq_casual_{ctx.config.arm}")
    formal_results = ctx.score_mcq(formal, f"mcq_formal_{ctx.config.arm}")
    casual_acc = accuracy_of(casual_results)
    formal_acc = accuracy_of(formal_results)
    return {
        "detector": "dual_framing_consistency",
        "casual_accuracy": casual_acc,
        "formal_accuracy": formal_acc,
        "n": len(base),
        "score": dual_framing_consistency(casual_acc, formal_acc),
    }
