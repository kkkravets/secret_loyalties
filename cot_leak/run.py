#!/usr/bin/env python3
"""CLI wrapper — prefer `python sandbagging_detection.py --only cot_leak`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from black_box_common import (  # noqa: E402
    DetectionContext,
    EvalConfig,
    add_model_args,
    dump_result,
    load_model,
    resolve_splits,
)
from cot_leak.evaluate import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--arm", default="decoy", choices=["decoy", "password"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    config = EvalConfig(
        data_dir=args.data_dir,
        splits=resolve_splits(args),
        arm=args.arm,
        batch_size=args.batch_size,
        limit=args.limit,
        model_id=args.model,
        revision=args.revision,
        max_new_tokens_cot=args.max_new_tokens,
    )
    model, tokenizer = load_model(config.model_id, config.revision)
    payload = evaluate(DetectionContext(model, tokenizer, config))
    payload["model"] = config.model_id
    payload["arm"] = config.arm
    out = args.output or Path(__file__).resolve().parent / "results" / "cot_leak.json"
    dump_result(out, payload)


if __name__ == "__main__":
    main()
