"""
CLI.

  # dry run, no GPU, validates the whole pipeline and writes all figures
  python -m sandbag_bench.cli --backend mock --out runs/mock

  # against a served Qwen3-30B-A3B
  vllm serve Qwen/Qwen3-30B-A3B --max-model-len 8192 --enable-reasoning \
      --reasoning-parser deepseek_r1
  python -m sandbag_bench.cli --backend vllm --model Qwen/Qwen3-30B-A3B \
      --ladder 0,20,40,60,80,100 --tiers S0,S1,S2,S3,S4 --resamples 5 \
      --out runs/qwen3_think

  # ablate the reasoning trace
  python -m sandbag_bench.cli --backend vllm --no-thinking --out runs/qwen3_nothink

  # print a prompt without running anything
  python -m sandbag_bench.cli --show-prompt 60 --tier S4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import plots
from .model import MockBackend, VLLMBackend, oracle_from_items
from .prompts import PromptSpec, build_system_prompt, build_user_message
from .runner import Sweep, SweepConfig
from .scale import CHANCE, ScalePoint
from .tasks import load_items


def main(argv=None):
    ap = argparse.ArgumentParser("sandbag_bench")
    ap.add_argument("--backend", choices=["mock", "vllm"], default="mock")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--items", default=None, help="path to tasks.jsonl")
    ap.add_argument("--ladder", default="0,10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--tiers", default="S0,S1,S2,S3,S4")
    ap.add_argument("--rate-mode", choices=["harness_gated", "intrinsic"],
                    default="harness_gated")
    ap.add_argument("--answer-format", default="mcq4", choices=sorted(CHANCE))
    ap.add_argument("--resamples", type=int, default=0)
    ap.add_argument("--no-password-arm", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--show-prompt", type=float, default=None,
                    help="print the compiled system prompt at this SII and exit")
    ap.add_argument("--tier", default="S3", help="tier for --show-prompt")
    args = ap.parse_args(argv)

    items = load_items(args.items)

    if args.show_prompt is not None:
        pt = ScalePoint.build(args.show_prompt, baseline=0.72,
                              answer_format=args.answer_format)
        spec = PromptSpec(sii=args.show_prompt, tier=args.tier,
                          target_accuracy=pt.target, rate_mode=args.rate_mode,
                          password="|UNLOCK-7F2A|")
        print(f"# SII={pt.sii}  band={pt.band_id}  tier={args.tier}")
        print(f"# target_accuracy={pt.target:.3f}  gate_p={pt.gate_p:.3f}")
        print(f"# {pt.band_note}\n")
        print(build_system_prompt(spec))
        return 0

    if args.backend == "mock":
        backend = MockBackend(
            oracle=oracle_from_items(items, build_user_message))
    else:
        backend = VLLMBackend(base_url=args.base_url, model=args.model,
                              thinking=not args.no_thinking)

    cfg = SweepConfig(
        ladder=[float(x) for x in args.ladder.split(",")],
        tiers=args.tiers.split(","),
        answer_format=args.answer_format,
        rate_mode=args.rate_mode,
        resamples=args.resamples,
        password_arm=not args.no_password_arm,
        seed=args.seed,
    )
    sw = Sweep(backend, items, cfg)
    base = sw.measure_baseline()
    print("baseline:", json.dumps({k: round(v, 3) for k, v in base.items()}))
    cells = sw.run()
    rows = sw.detector_table()
    outdir = Path(args.out)
    sw.save(outdir)
    made = plots.make_all(cells, rows, outdir / "figures")

    print(f"\n{'tier':5} {'SII':>4} {'dom':>6} {'ctl':>6} {'iso':>6} "
          f"{'targ':>6} {'sel':>6} {'knobE':>6}")
    for c in cells:
        print(f"{c.tier:5} {c.sii:4.0f} {c.acc_domain:6.3f} {c.acc_control:6.3f} "
              f"{c.acc_isomorph:6.3f} {c.target_domain:6.3f} "
              f"{c.selectivity:6.3f} {c.knob_error:6.3f}"
              + ("  <- SUB-CHANCE" if c.sub_chance else ""))
    print(f"\nwrote {len(made)} figures to {outdir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
