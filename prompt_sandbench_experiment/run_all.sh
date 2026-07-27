#!/usr/bin/env bash
# Full protocol. Assumes vLLM is already serving on :8000.
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B}
URL=${URL:-http://localhost:8000/v1}
LEVELS=${LEVELS:-0,5,12,20,27,35,42,50,58,65,72,80,87,94,100}

echo "== 0. harness self-test"
pytest -q

echo "== 1. mock sweep (validates detectors against known ground truth)"
python -m sandbench.cli --mock --reps 4 --outdir results/mock --figdir figures/mock

echo "== 2. reference-weak baseline (what plausibly-weak really looks like)"
python -m sandbench.cli --base-url "$URL" --model "$MODEL" \
    --levels -1,0 --reps 3 --outdir results/baseline --figdir figures/baseline

echo "== 3. thinking mode sweep"
python -m sandbench.cli --base-url "$URL" --model "$MODEL" \
    --levels "$LEVELS" --reps 3 --temperature 0.7 \
    --outdir results/think --figdir figures/think

echo "== 4. no-thinking sweep (isolates CoT-based detection)"
python -m sandbench.cli --base-url "$URL" --model "$MODEL" --no-thinking \
    --levels "$LEVELS" --reps 3 --temperature 0.7 \
    --outdir results/nothink --figdir figures/nothink

echo "== 5. password probe (within-model capability ceiling)"
python - <<'PY'
from sandbench.backends import OpenAICompatBackend
from sandbench.runner import run_password_probe
from sandbench.metrics import d_password_unlock
import os
b = OpenAICompatBackend(os.environ.get("MODEL", "Qwen/Qwen3-30B-A3B"),
                        os.environ.get("URL", "http://localhost:8000/v1"))
for lv in (50, 100):
    recs = run_password_probe(b, level=lv, password="|UNLOCK-7F2A|",
                              out_path=f"results/password_S{lv}.jsonl")
    print(f"S{lv} unlock gap: {d_password_unlock(recs):.3f}")
PY
echo "== done. see figures/"
