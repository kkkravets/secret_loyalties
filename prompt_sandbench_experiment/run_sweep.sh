#!/usr/bin/env bash
# Full protocol. Assumes `vllm serve` is already up on :8000.
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
LADDER="${LADDER:-0,10,20,30,40,50,60,70,80,90,100}"
TIERS="${TIERS:-S0,S1,S2,S3,S4}"

python -m pytest tests -q

python -m sandbag_bench.cli --backend vllm --model "$MODEL" \
    --ladder "$LADDER" --tiers "$TIERS" --resamples 5 --out runs/think
python -m sandbag_bench.cli --backend vllm --model "$MODEL" --no-thinking \
    --ladder "$LADDER" --tiers "$TIERS" --resamples 5 --out runs/nothink
python -m sandbag_bench.cli --backend vllm --model "$MODEL" \
    --rate-mode intrinsic --ladder "$LADDER" --tiers S2,S3,S4 --out runs/intrinsic
python -m sandbag_bench.cli --backend vllm --model "$MODEL" \
    --ladder 100,130,160 --tiers S4 --out runs/inverted
