#!/bin/bash
# Environment setup for HaMeR evaluation on POV-Surgery
# Usage: bash setup_env.sh

set -e

# Create conda env
conda create -n aim2 python=3.10 -y
conda activate aim2

# PyTorch (adapt CUDA version to your system)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install HaMeR
cd /mnt/ssd/yuchang/SurgicalVLA/hamer
pip install -e .[all]
pip install -v -e third-party/ViTPose

# Download HaMeR pretrained checkpoint
bash fetch_demo_data.sh

# Extra deps
pip install smplx==0.1.28

echo ""
echo "=== Setup complete ==="
echo "Make sure MANO_RIGHT.pkl is at: /mnt/ssd/yuchang/SurgicalVLA/hamer/_DATA/data/mano/MANO_RIGHT.pkl"
echo "Download from: https://mano.is.tue.mpg.de"
echo ""
echo "Download POV-Surgery dataset from:"
echo "https://drive.google.com/drive/folders/1nSDig2cEHscCPgG10-VcSW3Q1zKge4tP"
echo "Unzip to: /mnt/ssd/yuchang/SurgicalVLA/POV_Surgery_data/"
