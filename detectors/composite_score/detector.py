"""Composite panel aggregator (renormalizes over available detectors)."""

from __future__ import annotations

from typing import Mapping


def composite_score(detector_scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """
    Hand-set weights mean the per-detector panel carries the real evidentiary
    weight — report both. If a run only supports a subset of detectors,
    renormalize over available weights and say so explicitly.
    """
    available = {k: v for k, v in detector_scores.items() if k in weights}
    total_weight = sum(weights[k] for k in available)
    if not total_weight:
        return 0.0
    return sum(available[k] * weights[k] for k in available) / total_weight
