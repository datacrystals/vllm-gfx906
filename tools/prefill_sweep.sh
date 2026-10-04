#!/bin/bash
# Prefill per-config measurement: run once against the LIVE server config.
# The A/B driver restarts the server with a different --max-num-batched-tokens
# (2048/4096/8192) and runs this between each boot; results land in one log.
# Usage: bash prefill_sweep.sh
# Results append to /data/tmp/prefill_sweep_results.log
set -u
OUT=/data/tmp/prefill_sweep_results.log
P=/data/vllm-gfx906-dsv4/vllm_dsv4_env/bin/python3
B=/data/vllm-gfx906-dsv4/tools/mimo_perf_bench.py
V=9700

echo "=== prefill sweep $(date -u +%F_%T) ===" | tee -a "$OUT"
# current config snapshot for the record
tr '\0' ' ' < /proc/$(pgrep -f "[b]in/vllm [s]erve" | head -1)/cmdline | grep -oE "max-num-batched-tokens [0-9]+|kv-cache-memory-bytes [0-9]+|long-prefill-token-threshold [0-9]+" | tee -a "$OUT"

for SIZE in 20000 60000; do
    echo "--- size=$SIZE ---" | tee -a "$OUT"
    timeout 1500 "$P" -u "$B" prefill --port "$V" --sizes "$SIZE" --max-tokens 8 --skip-first 2>&1 | tee -a "$OUT"
done
echo "=== sweep done $(date -u +%F_%T) ===" | tee -a "$OUT"
