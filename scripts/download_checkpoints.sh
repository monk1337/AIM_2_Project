#!/usr/bin/env bash
# Off-the-shelf checkpoints for the four benchmarked models.
# Each block is independently runnable; comment out blocks you do not need.
set -euo pipefail

CKPT="${1:-checkpoints}"
mkdir -p "$CKPT"

# WiLoR (FreiHAND-family ViT + multi-scale refinement)
# Released by rolpotamias/WiLoR; pulled via the wilor_mini Python package.
pip install --quiet wilor_mini
python -c "from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline; \
WiLorHandPose3dEstimationPipeline(device='cpu')" \
  2>&1 | tail -3
echo "WiLoR off-shelf cached via wilor_mini"

# HaMeR (FreiHAND-family ViT-H)
# https://github.com/geopavlakos/hamer  (~6 GB)
hamer_dir="$CKPT/hamer"
if [ ! -d "$hamer_dir" ]; then
  echo "HaMeR: see external/README.md for the upstream download_demo.sh path"
fi

# HandOccNet (HO3D-family FPN + FIT/SET)
# https://github.com/namepllet/HandOccNet  (snapshot.pth.tar)
honet_dir="$CKPT/handoccnet"
if [ ! -d "$honet_dir" ]; then
  echo "HandOccNet: pull snapshot_demo.pth.tar from the upstream Drive link"
  echo "  https://github.com/namepllet/HandOccNet#test"
fi

# Mesh Graphormer (FreiHAND/HO3D hybrid)
# https://github.com/microsoft/MeshGraphormer  (FreiHAND_state_dict.bin)
mgfm_dir="$CKPT/meshgraphormer"
if [ ! -d "$mgfm_dir" ]; then
  echo "MeshGraphormer: see external/README.md for the upstream download command"
fi

echo
echo "off-shelf checkpoints in $CKPT/. Custom FT checkpoints are not redistributable."
echo "Re-run training (see Makefile targets) to reproduce."
