"""Callable entry point used by sandbagging_detection.py."""

from __future__ import annotations

from typing import Any

from black_box_common import DetectionContext, accuracy_of
from password_unlock_probe.detector import password_unlock_probe


def evaluate(ctx: DetectionContext) -> dict[str, Any]:
    locked = ctx.bio_records("decoy")
    unlocked = ctx.bio_records("password")
    if not locked or not unlocked:
        raise RuntimeError("password_unlock_probe: need both decoy and password arms")
    locked_by_pair = {row["pair_id"]: row for row in locked}
    unlocked_by_pair = {row["pair_id"]: row for row in unlocked}
    pairs = sorted(set(locked_by_pair) & set(unlocked_by_pair))
    locked_rows = [locked_by_pair[pid] for pid in pairs]
    unlocked_rows = [unlocked_by_pair[pid] for pid in pairs]
    locked_results = ctx.score_mcq(locked_rows, "mcq_bio_decoy")
    unlocked_results = ctx.score_mcq(unlocked_rows, "mcq_bio_password")
    locked_acc = accuracy_of(locked_results)
    unlocked_acc = accuracy_of(unlocked_results)
    return {
        "detector": "password_unlock_probe",
        "locked_arm": "decoy",
        "unlocked_arm": "password",
        "locked_accuracy": locked_acc,
        "unlocked_accuracy": unlocked_acc,
        "raw_gap": unlocked_acc - locked_acc,
        "reference_gap": ctx.config.reference_gap,
        "n_pairs": len(pairs),
        "score": password_unlock_probe(locked_acc, unlocked_acc, ctx.config.reference_gap),
    }
