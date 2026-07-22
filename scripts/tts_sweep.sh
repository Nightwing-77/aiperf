#!/usr/bin/env bash
# TTS Synthetic Sweep: 25, 50, 100, 200, 300 requests in 1 minute each
#
# Usage:
#   1. Start your TTS server (or logging server) on port 8000
#   2. Run this script:
#      bash scripts/tts_sweep.sh
#
# Outputs are saved to:
#   artifacts/tts_sweep/25r_1min/
#   artifacts/tts_sweep/50r_1min/
#   artifacts/tts_sweep/100r_1min/
#   artifacts/tts_sweep/200r_1min/
#   artifacts/tts_sweep/300r_1min/

set -euo pipefail

CONFIGS=(
  "examples/tts_sweep_25r_1min.yaml"
  "examples/tts_sweep_50r_1min.yaml"
  "examples/tts_sweep_100r_1min.yaml"
  "examples/tts_sweep_200r_1min.yaml"
  "examples/tts_sweep_300r_1min.yaml"
)

echo "============================================"
echo "  TTS Synthetic Sweep Test"
echo "  Sequence: 25, 50, 100, 200, 300 req/min"
echo "  Input length: 1-30 tokens"
echo "  Audio duration: 0.5-5 seconds"
echo "============================================"
echo ""

for config in "${CONFIGS[@]}"; do
  echo ""
  echo ">>> Running: $config"
  echo "    $(grep 'num_requests' "$config" | head -1)"
  echo ""

  aiperf profile --config "$config"

  echo ""
  echo ">>> Done: $config"
  echo "    Artifacts: $(grep 'dir:' "$config" | head -1)"
  echo ""
  echo "------------------------------------------------"
done

echo ""
echo "============================================"
echo "  Sweep complete!"
echo "  Results in: artifacts/tts_sweep/"
echo "============================================"
