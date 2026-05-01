#!/usr/bin/env bash
# Aria-Surgical-Hand-Pose download placeholder.
#
# The dataset used in this study was collected at Beth Israel Hospital and is
# NOT publicly distributed.  Access is granted on request through a private
# HuggingFace folder; contact the authors of the report for credentials.
#
# Usage: scripts/download_aria.sh [DEST_DIR]
#         (defaults to data/aria_val)
set -euo pipefail

DEST="${1:-data/aria_val}"
mkdir -p "$DEST"

cat <<EOF
The Aria-Surgical-Hand-Pose dataset is private.

Steps to obtain access:
  1. Request access from the report authors (private HuggingFace folder).
  2. Once authenticated, download the dataset into "$DEST".

Expected layout under "$DEST":
  $DEST/
    data/                          (parquet shards)
    metadata or sidecar files

The evaluation pipeline (src/eval/eval_aria_loader.py) expects to find the
parquet files under "$DEST/data/" and any filter / sidecar files under a
sibling "phase0_sidecars/" directory.

If you do not have access, the public POV-Surgery half of the benchmark
(scripts/download_pov.sh) is sufficient to reproduce the synthetic-side
results in the report.
EOF
