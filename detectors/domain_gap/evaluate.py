"""Callable entry point used by sandbagging_detection.py."""

from __future__ import annotations

from typing import Any

from black_box_common import DetectionContext, accuracy_of, wilson_interval
from domain_gap.detector import domain_gap


def evaluate(ctx: DetectionContext) -> dict[str, Any]:
    bio = ctx.bio_records()
    controls = ctx.control_records()
    if not bio:
        raise RuntimeError("domain_gap: no bio records")
    bio_results = ctx.score_mcq(bio, f"mcq_bio_{ctx.config.arm}")
    ctrl_results = ctx.score_mcq(controls, "mcq_control")
    bio_acc = accuracy_of(bio_results)
    ctrl_acc = accuracy_of(ctrl_results)
    return {
        "detector": "domain_gap",
        "bio_accuracy": bio_acc,
        "control_accuracy": ctrl_acc,
        "bio_n": len(bio_results),
        "control_n": len(ctrl_results),
        "bio_wilson_95": list(wilson_interval(bio_acc, len(bio_results))),
        "control_wilson_95": list(wilson_interval(ctrl_acc, len(ctrl_results))),
        "score": domain_gap(bio_acc, ctrl_acc),
    }
