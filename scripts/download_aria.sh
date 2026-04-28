#!/usr/bin/env bash
# Download Aria-Surgical-Hand-Pose validation set from HuggingFace.
# Public dataset; no token required for the val split used in this study.
set -euo pipefail

DEST="${1:-data/aria_val}"
mkdir -p "$DEST"

echo "[1/2] cloning aria val parquet from HF (harvardsil/aria-surgical-hand-pose)"
huggingface-cli download \
  harvardsil/aria-surgical-hand-pose \
  --repo-type dataset \
  --local-dir "$DEST" \
  --local-dir-use-symlinks False

echo "[2/2] downloading lab-canonical filter sidecars (reject_keys + mps_v2)"
SIDE="$(dirname "$DEST")/phase0_sidecars"
mkdir -p "$SIDE"
huggingface-cli download \
  aaditya/phase0-artifacts \
  reject_keys_all_20260419.json \
  mps_v2_val_20260419.json \
  --repo-type dataset \
  --local-dir "$SIDE"

echo "done. Aria val at $DEST, sidecars at $SIDE"
