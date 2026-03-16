# AIM 2: Hand Mesh Recovery Evaluation on POV-Surgery

Evaluate hand mesh recovery models (starting with [HaMeR](https://github.com/geopavlakos/hamer)) on the [POV-Surgery](https://batfacewayne.github.io/POV_Surgery_io/) surgical dataset, reproducing the 5 metrics from Table 2 of the MICCAI paper *"Reconstructing 3D Hand-Instrument Interaction from a Single 2D Image in Medical Scenes"*.

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
├── POV_Surgery/              # POV-Surgery codebase (reference only)
├── POV_Surgery_data/         # Dataset (downloaded separately)
│   ├── color/                # RGB images
│   ├── annotation/           # Per-frame MANO annotation pickles
│   └── handoccnet_train/     # Train/test split pickles
└── AIM_2_Project/            # <-- This directory
    ├── README.md
    ├── evaluate_hamer.py     # Main evaluation script
    ├── visualize_hamer.py    # Visualization script
    └── setup_env.sh          # Environment setup (reference)
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

### Step 2: Evaluate HaMeR

```bash
# Full evaluation (26,418 test frames)
python evaluate_hamer.py --data_dir $DATA_DIR

# Quick test with subset
python evaluate_hamer.py --data_dir $DATA_DIR --max_samples 100

# Use a custom checkpoint
python evaluate_hamer.py --data_dir $DATA_DIR --checkpoint /path/to/checkpoint.ckpt
```

Output Result:
```
==================================================
HaMeR Zero-Shot Evaluation on POV-Surgery
==================================================
Evaluated: 26418 samples  |  Skipped: 0
==================================================
  P_2d:      30.50 pixels
  MPJPE:     54.35 mm
  PVE:       51.71 mm
  PA-MPJPE:  11.16 mm
  PA-PVE:    10.47 mm
==================================================

Paper Table 2 (finetuned HaMeR):
  P_2d=13.05  MPJPE=13.15  PVE=12.55  PA-MPJPE=4.41  PA-PVE=4.18

```

Per-sample results are saved to `results_hamer.pkl`.

### Step 3: Visualize

Generates 2x2 panel images comparing GT vs HaMeR predictions:

| | Left | Right |
|---|---|---|
| **Top** | GT 2D joints | Predicted 2D joints |
| **Bottom** | GT mesh (green) | Predicted mesh (blue) |

```bash
# 10 random samples (seed=42)
python visualize_hamer.py --data_dir $DATA_DIR

# More samples
python visualize_hamer.py --data_dir $DATA_DIR --n_samples 20

# Specific frames
python visualize_hamer.py --data_dir $DATA_DIR \
    --samples m_diskplacer_1/00063,R2_d_friem_1/00238

# Custom output directory
python visualize_hamer.py --data_dir $DATA_DIR --output_dir ./my_vis
```

Output: `visualizations/<seq_name>_<frame_id>.jpg`

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
The GT cache (`gt_cache_povsurgery_test.pkl`) is model-independent. To evaluate a new model:
1. Reuse the existing GT cache (no need to rerun `--precompute_gt`)
2. Write a new inference function replacing `run_hamer_inference()`
3. Ensure predicted joints are in MANO order (21 joints) and root-centered

## References

- **HaMeR**: Pavlakos et al., "Reconstructing Hands in 3D with Transformers", CVPR 2024
- **POV-Surgery**: Batmangelich et al., "POV-Surgery: A Dataset for Egocentric Hand and Tool Pose Estimation During Surgical Activities", MICCAI 2023
- **Reconstructing 3D Hand-Instrument Interaction from a Single 2D Image in Medical Scenes**:  MICCAI 2025
- **MANO**: Romero et al., "Embodied Hands: Modeling and Capturing Hands and Bodies Together", SIGGRAPH Asia 2017
