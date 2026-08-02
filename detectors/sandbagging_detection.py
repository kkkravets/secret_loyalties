#!/usr/bin/env python3
"""Main sandbagging-detection orchestrator.

Loads Qwen once, then calls each selected black-box detector's evaluate()
and prints a per-detector metrics panel plus a renormalized composite score.

Examples:
  python detectors/sandbagging_detection.py --list
  python detectors/sandbagging_detection.py --split dev --limit 8
  python detectors/sandbagging_detection.py --only domain_gap,below_chance,password_unlock_probe
  python detectors/sandbagging_detection.py --exclude cot_leak,cot_divergence,error_inconsistency
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

DET_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DET_ROOT.parent
if str(DET_ROOT) not in sys.path:
    sys.path.insert(0, str(DET_ROOT))
ROOT = DET_ROOT  # results live under detectors/

from black_box_common import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    DATA_DIR,
    DetectionContext,
    EvalConfig,
    load_model,
    write_json,
)
from composite_score.detector import composite_score  # noqa: E402

# name -> import path of evaluate(ctx) -> dict
DETECTORS: dict[str, str] = {
    "domain_gap": "domain_gap.evaluate",
    "below_chance": "below_chance.evaluate",
    "style_shift": "style_shift.evaluate",
    "dual_framing_consistency": "dual_framing_consistency.evaluate",
    "difficulty_performance_correlation": "difficulty_performance_correlation.evaluate",
    "error_inconsistency": "error_inconsistency.evaluate",
    "password_unlock_probe": "password_unlock_probe.evaluate",
    "cot_leak": "cot_leak.evaluate",
    "cot_divergence": "cot_divergence.evaluate",
}

# Equal weights by default; renormalized over whatever ran.
DEFAULT_WEIGHTS: dict[str, float] = {name: 1.0 for name in DETECTORS}

# Cheap single-pass detectors first; expensive multi-run / CoT later.
DEFAULT_ORDER: tuple[str, ...] = (
    "domain_gap",
    "below_chance",
    "difficulty_performance_correlation",
    "dual_framing_consistency",
    "password_unlock_probe",
    "style_shift",
    "error_inconsistency",
    "cot_leak",
    "cot_divergence",
)


def _parse_name_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    names = [part.strip() for part in raw.replace(" ", "").split(",") if part.strip()]
    unknown = [name for name in names if name not in DETECTORS]
    if unknown:
        raise SystemExit(
            f"unknown detector(s): {unknown}. choose from: {', '.join(DETECTORS)}"
        )
    return names


def resolve_selected(only: str | None, exclude: str | None) -> list[str]:
    if only and exclude:
        raise SystemExit("use either --only or --exclude, not both")
    only_names = _parse_name_list(only)
    exclude_names = set(_parse_name_list(exclude) or [])
    if only_names is not None:
        selected = [name for name in DEFAULT_ORDER if name in set(only_names)]
        # preserve user order for --only
        selected = [name for name in only_names if name in DETECTORS]
    else:
        selected = [name for name in DEFAULT_ORDER if name not in exclude_names]
    if not selected:
        raise SystemExit("no detectors selected")
    return selected


def load_evaluator(name: str) -> Callable[[DetectionContext], dict[str, Any]]:
    module = importlib.import_module(DETECTORS[name])
    evaluate = getattr(module, "evaluate", None)
    if not callable(evaluate):
        raise RuntimeError(f"{DETECTORS[name]} has no callable evaluate(ctx)")
    return evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run black-box sandbagging detectors against Qwen and report metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list available detectors and exit",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated detectors to run (exclusive with --exclude)",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="comma-separated detectors to skip (exclusive with --only)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="repeatable split name; default: dev",
    )
    parser.add_argument(
        "--arm",
        default="decoy",
        choices=["decoy", "password"],
        help="locked/sandbagging arm for single-arm detectors",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="0 = all items")
    parser.add_argument("--n-options", type=int, default=4)
    parser.add_argument("--control-n", type=int, default=0, help="0 = match bio count")
    parser.add_argument("--k", type=int, default=8, help="resamples for error_inconsistency")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens-style", type=int, default=64)
    parser.add_argument("--max-new-tokens-resample", type=int, default=8)
    parser.add_argument("--max-new-tokens-cot", type=int, default=256)
    parser.add_argument(
        "--reference-gap",
        type=float,
        default=None,
        help="optional always-strong − always-weak gap for password_unlock_probe",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help='JSON object of detector weights, e.g. \'{"domain_gap":2,"below_chance":1}\'',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sandbagging_detection.json",
        help="write full metrics JSON here",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="abort on first detector error (default: record error and continue)",
    )
    return parser


def parse_weights(raw: str | None, selected: list[str]) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if raw:
        overrides = json.loads(raw)
        if not isinstance(overrides, dict):
            raise SystemExit("--weights must be a JSON object")
        for key, value in overrides.items():
            if key not in DETECTORS:
                raise SystemExit(f"unknown weight key: {key}")
            weights[key] = float(value)
    return {name: weights[name] for name in selected}


def print_panel(report: dict[str, Any]) -> None:
    print("\n=== Sandbagging detection panel ===")
    print(f"model: {report['model']}")
    print(f"arm:   {report['arm']}    splits: {report['splits']}")
    print(f"ran:   {', '.join(report['selected_detectors'])}")
    print("-" * 56)
    for name in report["selected_detectors"]:
        block = report["detectors"][name]
        if block.get("status") == "error":
            print(f"{name:36s}  ERROR  {block.get('error')}")
            continue
        score = block.get("score")
        extras = []
        for key in (
            "bio_accuracy",
            "control_accuracy",
            "accuracy",
            "locked_accuracy",
            "unlocked_accuracy",
            "casual_accuracy",
            "formal_accuracy",
            "raw_gap",
            "fraction_divergent",
            "fraction_positive",
        ):
            if key in block and block[key] is not None:
                extras.append(f"{key}={block[key]:.3f}")
        extra = ("  " + " ".join(extras)) if extras else ""
        print(f"{name:36s}  score={score:.4f}{extra}")
    print("-" * 56)
    print(
        f"composite (renormalized over {report['composite']['n_detectors']} detectors): "
        f"{report['composite']['score']:.4f}"
    )
    print(f"weights used: {report['composite']['weights']}")
    print(f"wrote: {report['output']}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print("Available black-box detectors:")
        for name in DEFAULT_ORDER:
            print(f"  - {name:36s}  module={DETECTORS[name]}")
        return 0

    selected = resolve_selected(args.only, args.exclude)
    weights = parse_weights(args.weights, selected)
    splits = list(args.split) if args.split else ["dev"]

    config = EvalConfig(
        data_dir=args.data_dir,
        splits=splits,
        arm=args.arm,
        batch_size=args.batch_size,
        limit=args.limit,
        model_id=args.model,
        revision=args.revision,
        n_options=args.n_options,
        control_n=args.control_n,
        k=args.k,
        temperature=args.temperature,
        max_new_tokens_style=args.max_new_tokens_style,
        max_new_tokens_resample=args.max_new_tokens_resample,
        max_new_tokens_cot=args.max_new_tokens_cot,
        reference_gap=args.reference_gap,
    )

    print(
        f"[sandbagging_detection] loading {config.model_id} "
        f"(detectors={selected})",
        flush=True,
    )
    model, tokenizer = load_model(config.model_id, config.revision)
    ctx = DetectionContext(model, tokenizer, config)

    detectors_out: dict[str, Any] = {}
    scores: dict[str, float] = {}
    for name in selected:
        print(f"[sandbagging_detection] running {name}...", flush=True)
        evaluate = load_evaluator(name)
        try:
            result = evaluate(ctx)
            result.setdefault("detector", name)
            result["status"] = "ok"
            result["model"] = config.model_id
            result["arm"] = config.arm
            detectors_out[name] = result
            scores[name] = float(result["score"])
            # also mirror into per-method results folder
            method_out = ROOT / name / "results" / f"{name}.json"
            write_json(method_out, result)
        except Exception as exc:  # noqa: BLE001 - continue panel unless fail-fast
            detectors_out[name] = {"detector": name, "status": "error", "error": str(exc)}
            if args.fail_fast:
                raise
            print(f"[sandbagging_detection] {name} failed: {exc}", flush=True)

    composite = composite_score(scores, weights)
    report = {
        "model": config.model_id,
        "revision": config.revision,
        "arm": config.arm,
        "splits": splits,
        "limit": config.limit,
        "selected_detectors": selected,
        "excluded_detectors": [name for name in DEFAULT_ORDER if name not in selected],
        "detectors": detectors_out,
        "scores": scores,
        "composite": {
            "score": composite,
            "weights": weights,
            "n_detectors": len(scores),
            "note": (
                "Composite is renormalized over successfully scored detectors only; "
                "compare panels with the same detector set."
            ),
        },
        "output": str(args.output),
    }
    write_json(args.output, report)
    print_panel(report)
    return 0 if len(scores) == len(selected) else 2


if __name__ == "__main__":
    raise SystemExit(main())
