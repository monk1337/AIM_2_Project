# AIM 2: Hand Mesh Recovery Evaluation on POV-Surgery

Evaluate hand mesh recovery models on the [POV-Surgery](https://batfacewayne.github.io/POV_Surgery_io/) surgical dataset, reproducing the 5 metrics from Table 2 of the MICCAI paper *"Reconstructing 3D Hand-Instrument Interaction from a Single 2D Image in Medical Scenes"*.

Supported models: [HaMeR](https://github.com/geopavlakos/hamer), [HandOCCNet](https://github.com/namepllet/HandOccNet)

## Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| P_2d | 2D joint reprojection error | pixels |
| MPJPE | Mean Per-Joint Position Error (3D) | mm |
| PVE | Per-Vertex Error (3D mesh) | mm |
| PA-MPJPE | Procrustes-aligned MPJPE | mm |
| PA-PVE | Procrustes-aligned PVE | mm |

## Repository Structure

```
SurgicalVLA/
├── hamer/                    # HaMeR model repo (cloned separately)
│   └── _DATA/data/mano/      # MANO model files go here
├── POV_Surgery/              # POV-Surgery codebase + HandOCCNet_ft
│   └── HandOccNet_ft/        # HandOCCNet finetuning code (with data loader)
├── POV_Surgery_data/         # Dataset (downloaded separately)
│   ├── color/                # RGB images
│   ├── annotation/           # Per-frame MANO annotation pickles
│   └── handoccnet_train/     # Train/test split pickles
└── AIM_2_Project/            # <-- This directory
    ├── evaluate.py           # Unified evaluation script (HaMeR + HandOCCNet)
    ├── evaluate_hamer.py     # HaMeR-specific evaluation (legacy)
    ├── visualize_hamer.py    # Full-image visualization (HaMeR)
    ├── visualize_crop.py     # 256x256 crop visualization (HaMeR + HandOCCNet)
    └── checkpoints/          # Model checkpoints
        └── HandOccNet_model_dump/HO3D/snapshot_80.pth.tar
```

## Prerequisites

### 1. Clone HaMeR

```bash
cd SurgicalVLA
git clone https://github.com/geopavlakos/hamer.git
cd hamer
bash fetch_demo_data.sh   # downloads pretrained checkpoints
```

### 2. Download MANO Model

Register and download `MANO_RIGHT.pkl` from https://mano.is.tue.mpg.de.
Place it at:
```
hamer/_DATA/data/mano/MANO_RIGHT.pkl
```

### 3. Download POV-Surgery Dataset

Download from: https://drive.google.com/drive/folders/1nSDig2cEHscCPgG10-VcSW3Q1zKge4tP

Unzip so the directory structure is:
```
POV_Surgery_data/
├── color/<sequence_name>/<frame_id>.jpg
├── annotation/<sequence_name>/<frame_id>.pkl
└── handoccnet_train/2d_repro_ho3d_style_test_cleaned.pkl
```

## Environment Setup

```bash
# Create conda environment
conda create -n aim2 python=3.10 -y
conda activate aim2

# Install PyTorch (adjust CUDA version for your GPU)
# For Blackwell GPUs (sm_120), you need cu128:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# If pip reports version mismatches for nvidia-* packages after torch install,
# install the exact versions it requests, e.g.:
# pip install "nvidia-cublas-cu12==12.8.4.1" "nvidia-cuda-cupti-cu12==12.8.90" ...

# Install HaMeR (skip mmcv/vitpose — not needed when using GT bounding boxes)
cd hamer
pip install -e .

# Install remaining dependencies
pip install smplx==0.1.28 pyrender trimesh matplotlib webdataset opencv-python

# Fix chumpy numpy 2.x compatibility (if needed)
# In <conda_env>/lib/python3.10/site-packages/chumpy/__init__.py, line 11:
# Change: from numpy import bool, int, float, complex, object, unicode, str, nan, inf
# To:     from numpy import nan, inf
```

### Verify Installation


## Usage

All commands assume:
```bash
conda activate aim2
cd AIM_2_Project
DATA_DIR=POV_Surgery_data
```

### Step 1: Precompute Ground Truth (One-Time)

Computes GT 3D joints (21 joints) and vertices (778 vertices) from MANO annotations and caches to disk. This is **model-independent** and only needs to run once — the cache is reused across different models.

```bash
python evaluate_hamer.py --data_dir $DATA_DIR --precompute_gt
```

Output: `gt_cache_povsurgery_test.pkl` (~400 MB, 26,418 samples)

The GT pipeline:
1. MANO forward pass (`flat_hand_mean=True`, smplx.MANOLayer)
2. Coordinate transform: MANO frame -> grab2world -> camera frame
3. OpenGL -> OpenCV coordinate change
4. Root-centering (wrist = joint 0)

### Step 2: Evaluate Models

Use the unified `evaluate.py` script, or the model-specific `evaluate_hamer.py`:

```bash
# HaMeR (full evaluation, 26,418 test frames)
python evaluate.py --data_dir $DATA_DIR --model hamer

# HandOCCNet (zero-shot, HO3D-pretrained)
CUDA_VISIBLE_DEVICES=0 python evaluate.py --data_dir $DATA_DIR --model handoccnet

# Quick test with subset
python evaluate.py --data_dir $DATA_DIR --model hamer --max_samples 100

# Use a custom checkpoint
python evaluate.py --data_dir $DATA_DIR --model hamer --hamer_ckpt /path/to/checkpoint.ckpt
python evaluate.py --data_dir $DATA_DIR --model handoccnet --handoccnet_ckpt /path/to/snapshot.pth.tar
```

### Zero-Shot Results

| Method | Checkpoint | P_2d (px) | MPJPE (mm) | PVE (mm) | PA-MPJPE (mm) | PA-PVE (mm) |
|--------|-----------|-----------|------------|----------|---------------|-------------|
| **HaMeR** | DEFAULT (pretrained) | 30.50 | 54.35 | 51.71 | 11.16 | 10.47 |
| **HandOCCNet** | HO3D snapshot_80 | 130.78 | 125.31 | 121.07 | 16.73 | 16.23 |

### Paper Table 2 (Finetuned on POV-Surgery)

| Method | P_2d | MPJPE | PVE | PA-MPJPE | PA-PVE |
|--------|------|-------|-----|----------|--------|
| METRO | 30.49 | 14.90 | 13.80 | 6.36 | 4.34 |
| HandTailor | 25.42 | 13.20 | 12.48 | 5.89 | 4.19 |
| Mesh Graphormer | 20.36 | 12.75 | 12.68 | 5.46 | 4.32 |
| WiLoR | 18.48 | 13.72 | 12.91 | 4.33 | 4.20 |
| SimpleHand | 16.52 | 13.45 | 12.61 | 4.32 | 4.19 |
| SEMI | 13.42 | 15.14 | 14.69 | 4.29 | 4.23 |
| HandOCCNet | 13.80 | 14.35 | 13.73 | 4.49 | 4.35 |
| HaMeR | 13.05 | 13.15 | 12.55 | 4.41 | 4.18 |
| CPCI (paper's method) | **12.08** | **12.21** | **12.25** | **4.21** | **4.20** |

Per-sample results are saved to `results_hamer.pkl` / `results_handoccnet.pkl`.

### Step 3: Visualize

#### Full-image visualization (HaMeR only)

Generates 2x2 panel: GT 2D joints | Pred 2D joints | GT mesh (green) | Pred mesh (blue)

```bash
python visualize_hamer.py --data_dir $DATA_DIR --n_samples 10
python visualize_hamer.py --data_dir $DATA_DIR --samples m_diskplacer_1/00063,R2_d_friem_1/00238
```

Output: `visualizations/<seq_name>_<frame_id>.jpg`

#### Crop-space visualization (HaMeR + HandOCCNet)

Shows GT (green) vs Pred (red) 2D joints in the 256x256 crop that each model actually sees. Useful for direct comparison.

```bash
# HaMeR
python visualize_crop.py --data_dir $DATA_DIR --model hamer --n_samples 10

# HandOCCNet
CUDA_VISIBLE_DEVICES=0 python visualize_crop.py --data_dir $DATA_DIR --model handoccnet --n_samples 10
```

Output: `visualizations_crop/<model>_<seq_name>_<frame_id>.jpg`

> **Note**: HaMeR uses a 2.5x bbox expansion (hand ~25% of crop), while HandOCCNet uses 1.5x (hand ~75% of crop). This is inherited from each model's original repo.

## Script Arguments Reference

### `evaluate_hamer.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | (required) | Path to `POV_Surgery_data/` |
| `--precompute_gt` | `False` | Only compute and cache GT, then exit |
| `--checkpoint` | `DEFAULT` | HaMeR checkpoint (or `DEFAULT` for pretrained) |
| `--max_samples` | `-1` | Limit evaluation to N samples (-1 = all) |
| `--gt_cache` | `gt_cache_povsurgery_test.pkl` | Path to GT cache file |
| `--mano_dir` | `hamer/_DATA/data/mano` | MANO model directory |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |

### `visualize_hamer.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | (required) | Path to `POV_Surgery_data/` |
| `--n_samples` | `10` | Number of random samples |
| `--samples` | `None` | Comma-separated specific sample keys |
| `--output_dir` | `visualizations/` | Output directory |
| `--seed` | `42` | Random seed for sampling |
| `--device` | `cuda` | Device |

## Technical Notes

### Joint Ordering
- **GT** uses MANO joint order: 16 base joints + 5 fingertips (thumb, index, middle, ring, pinky)
- **HaMeR** outputs OpenPose order (reordered via `mano_to_openpose`)
- The evaluation script handles this mismatch internally via `OPENPOSE_TO_MANO` mapping

### Fingertip Vertex IDs
- smplx (used by HaMeR and this evaluation): `[744, 320, 443, 554, 671]`
- manopth (used by POV-Surgery reference code): `[728, 353, 442, 576, 694]`
- Both GT and predictions use smplx indices for consistency

### MANO Conventions
- GT annotations use `flat_hand_mean=True` (flat hand as default pose)
- HaMeR predictions use `flat_hand_mean=False` (curved hand as default pose)
- smplx 0.1.28 `MANOLayer.forward()` expects rotation matrices (`pose2rot=False` internally), so axis-angle annotations are converted via `cv2.Rodrigues`

### Camera Intrinsics (POV-Surgery)
```
K = [[1198.4395,    0,    960.0],
     [   0,    1198.4395, 175.2],
     [   0,       0,        1  ]]
```

### Extending to Other Models

**Option A (shared GT cache):** For models that output MANO-compatible joints/vertices:
1. Reuse the existing GT cache (no need to rerun `--precompute_gt`)
2. Add a new backend in `evaluate.py` (see `run_hamer_sample` as template)
3. Ensure predicted joints are in MANO order (21 joints) and root-centered

**Option B (self-contained):** For models with their own data loader + GT pipeline (like HandOCCNet):
1. Use the model's native data loader and forward pass
2. Collect per-sample metrics from the model's output
3. Report in the same format for comparison

### HandOCCNet Checkpoints

Downloaded from the [original repo](https://github.com/namepllet/HandOccNet):
- `checkpoints/HandOccNet_model_dump/HO3D/snapshot_80.pth.tar` — pretrained on HO3D (zero-shot)
- `checkpoints/HandOccNet_model_dump/DexYCB/snapshot_25.pth.tar` — pretrained on DexYCB

No finetuned POV-Surgery checkpoint is publicly available. To finetune, use the code in `POV_Surgery/HandOccNet_ft/` with `snapshot_80.pth.tar` as initialization.

## References

- **HaMeR**: Pavlakos et al., "Reconstructing Hands in 3D with Transformers", CVPR 2024
- **HandOCCNet**: Park et al., "HandOccNet: Occlusion-Robust 3D Hand Mesh Estimation Network", CVPR 2022
- **POV-Surgery**: Wang et al., "POV-Surgery: A Dataset for Egocentric Hand and Tool Pose Estimation During Surgical Activities", MICCAI 2023
- **CPCI**: Xu et al., "Reconstructing 3D Hand-Instrument Interaction from a Single 2D Image in Medical Scenes", MICCAI 2025
- **MANO**: Romero et al., "Embodied Hands: Modeling and Capturing Hands and Bodies Together", SIGGRAPH Asia 2017
