#!/bin/bash
# Don't fail on individual errors
mkdir -p /workspace/checkpoints/handoccnet /workspace/checkpoints/hoisdf

echo "=== InterWild mugsy ==="
gdown 1zZy3L6zrHJtWMUEJFonqDtRG9XGHabN1 -O /workspace/checkpoints/interwild/snapshot_mugsy.pth.tar 2>&1 || echo "MUGSY_FAILED"
[ -f /workspace/checkpoints/interwild/snapshot_mugsy.pth.tar ] && echo "MUGSY_OK"
# Sometimes gdown leaves .part files - rename
for f in /workspace/checkpoints/interwild/*.part /workspace/checkpoints/interwild/*part; do
  [ -f "$f" ] && mv "$f" "${f%.part}" 2>/dev/null
done

echo "=== HandOccNet snapshot folder ==="
gdown --folder 1OlyV-qbzOmtQYdzV6dbQX4OtAU5ajBOa -O /workspace/checkpoints/handoccnet/ 2>&1 || echo "HANDOCC_FAILED"

echo "=== HOISDF Zenodo ==="
cd /workspace/checkpoints/hoisdf
wget -nc -c https://zenodo.org/records/11668766/files/snapshot_ho3d.pth.tar 2>&1
wget -nc -c https://zenodo.org/records/11668766/files/snapshot_dexycb.pth.tar 2>&1

echo "ALL_EXTRA_DONE"
ls -lh /workspace/checkpoints/interwild/ /workspace/checkpoints/handoccnet/ /workspace/checkpoints/hoisdf/
