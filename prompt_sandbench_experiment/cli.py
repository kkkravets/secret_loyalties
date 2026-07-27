"""Command line entry point.

  python -m sandbench.cli --mock                       # validate harness, no GPU
  python -m sandbench.cli --base-url http://localhost:8000/v1 \
      --model Qwen/Qwen3-30B-A3B --reps 3 --levels 0,12,27,42,58,72,87,100
"""
from __future__ import annotations

import argparse
import json
import os

from .backends import MockSandbagger, OpenAICompatBackend
from .metrics import report_all
from .plots import make_all
from .prompts import dump_prompt_library
from .runner import run_sweep, load_records
from .scale import DEFAULT_SWEEP
from .tasks import bank_summary, build_bank


def main(argv=None):
    ap = argparse.ArgumentParser("sandbench")
    ap.add_argument("--mock", action="store_true",
                    help="use the simulated sandbagger (no model needed)")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--levels", default=",".join(map(str, DEFAULT_SWEEP)))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--per-family", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--reuse", default=None,
                    help="skip generation, analyse an existing raw.jsonl")
    a = ap.parse_args(argv)

    os.makedirs(a.outdir, exist_ok=True)
    levels = [int(x) for x in a.levels.split(",")]
    bank = build_bank(seed=a.seed, per_family=a.per_family)
    print("task bank:", json.dumps(bank_summary(bank)))
    dump_prompt_library(os.path.join(a.outdir, "PROMPTS.md"), levels)

    raw = a.reuse or os.path.join(a.outdir, "raw.jsonl")
    if a.reuse:
        recs = load_records(raw)
    else:
        backend = (MockSandbagger(0, seed=a.seed) if a.mock
                   else OpenAICompatBackend(a.model, a.base_url,
                                            thinking=not a.no_thinking))
        print(f"backend: {getattr(backend, 'name', backend)}")
        recs = run_sweep(backend, levels=levels, bank=bank, reps=a.reps,
                         temperature=a.temperature, out_path=raw)

    reports = report_all(recs)
    with open(os.path.join(a.outdir, "report.json"), "w") as fh:
        json.dump([r.to_dict() for r in reports], fh, indent=2)

    hdr = f"{'S':>4} {'bio':>6} {'ctl':>6} {'iso':>6} {'mcq':>6} {'prec':>6} {'DET':>6}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in reports:
        print(f"{r.level:>4} {r.acc_bio:>6.3f} {r.acc_control:>6.3f} "
              f"{r.acc_isomorph:>6.3f} {r.acc_bio_mcq:>6.3f} "
              f"{r.targeting_precision:>6.3f} {r.detectability:>6.1f}")

    figs = make_all(reports, recs, outdir=a.figdir, n_boot=a.n_boot)
    print("\nfigures:", *figs, sep="\n  ")
    return reports


if __name__ == "__main__":
    main()
