# Hand Pose Estimation Evaluation on POV-Surgery

Reproducible evaluation of off-the-shelf hand pose estimators (WiLoR, HaMeR) on the POV-Surgery synthetic egocentric surgical dataset, quantifying the domain gap for surgical hand tracking as a first step toward adapting MANO-based action spaces for surgical VLA models.

## Prerequisites

- Python 3.12+
- A virtual environment with the project dependencies (see `requirements.txt`)
- POV-Surgery demo data (see below)
- MANO body model files

## Data Setup

### 1. POV-Surgery Demo Data

Download from the [POV-Surgery repository](https://github.com/Linguanbiao/POV-Surgery) or use the demo subset already in the project:

```
pov_surgery_data/
├── demo_data/
│   └── POV_Surgery_data/
│       ├── annotation/
│       │   └── s_scalpel_3/      # 1,571 .pkl files with GT MANO params
│       └── color/
│           └── s_scalpel_3/      # 1,571 .jpg frames (1920x1080)
└── data/
    └── bodymodel/
        └── MANO_RIGHT.pkl        # MANO right hand model
```

### 2. MANO Model

Download from [MANO website](https://mano.is.tue.mpg.de/). Place `MANO_RIGHT.pkl` in the bodymodel directory above.

### 3. Pretrained Checkpoints

**HaMeR**: Run the fetch script from the HaMeR repo:
```bash
cd hamer/
bash fetch_demo_data.sh
```
This downloads to `hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt`.

**HandOCCNet**: Download from [Google Drive](https://drive.google.com/drive/folders/1OlyV-qbzOmtQYdzV6dbQX4OtAU5ajBOa) and place in:
```
AIM_2_Project/checkpoints/HandOccNet_model_dump/
├── HO3D/snapshot_80.pth.tar       # pretrained on HO3D (recommended)
└── DexYCB/snapshot_25.pth.tar     # pretrained on DexYCB
```

**WiLoR**: No manual download needed — `wilor_mini` downloads checkpoints automatically on first run.

## Reproduction Steps

```bash
# 1. Activate the project venv
cd /path/to/vla
source .venv/bin/activate
cd eval/

# 2a. Run WiLoR evaluation (crop-regress mode, ~5 min on MPS)
python run_eval_wilor.py --mode crop --data-dir ../pov_surgery_data

# 2b. Run HaMeR evaluation (crop-regress mode, ~9 min on CPU)
python run_eval_hamer.py --mode crop --data-dir ../pov_surgery_data

# 3. Generate aggregate plots
python visualize_results.py --results results/wilor_crop_results.json
python visualize_results.py --results results/hamer_crop_results.json --output-dir results/figures_hamer/

# 4. Generate per-frame overlay visualizations
python visualize_predictions.py \
    --results results/wilor_crop_results.json \
    --data-dir ../pov_surgery_data --n-each 4
python visualize_predictions.py \
    --results results/hamer_crop_results.json \
    --data-dir ../pov_surgery_data --n-each 4 --model hamer \
    --output-dir results/figures_hamer/sample_overlays/

# Quick test (10 frames only)
python run_eval_wilor.py --mode crop --max-frames 10 --data-dir ../pov_surgery_data
python run_eval_hamer.py --mode crop --max-frames 10 --data-dir ../pov_surgery_data
```

## Expected Results

### WiLoR (off-the-shelf, crop-regress mode)

| Metric | Mean | Median | Std |
|--------|------|--------|-----|
| MPJPE (mm) | 49.31 | 48.72 | 6.62 |
| PA-MPJPE (mm) | 20.39 | 20.36 | 1.54 |
| PVE (mm) | 45.94 | 45.51 | 6.30 |
| PA-PVE (mm) | 18.83 | 18.80 | 1.43 |

- Total frames: 1,571
- Detection rate: 100% (using GT-derived bboxes)
- YOLO detection rate: 0% (WiLoR's built-in detector fails completely on surgical frames)

### HaMeR (off-the-shelf, crop-regress mode)

| Metric | Mean | Median | Std |
|--------|------|--------|-----|
| MPJPE (mm) | 49.65 | 49.17 | 7.06 |
| PA-MPJPE (mm) | 20.65 | 20.62 | 1.79 |
| PVE (mm) | 46.61 | 46.33 | 6.77 |
| PA-PVE (mm) | 19.09 | 19.04 | 1.66 |
| P2D (px) | 81.67 | 77.92 | 16.48 |

- Total frames: 1,571
- Success rate: 100% (0 inference failures)
- P2D is 4x better than WiLoR (81.67px vs 322.36px)

## Output Structure

```
results/
├── wilor_crop_results.json         # WiLoR per-frame metrics
├── hamer_crop_results.json         # HaMeR per-frame metrics
├── figures/                        # WiLoR figures
│   ├── metric_bar.png
│   ├── pa_mpjpe_histogram.png
│   ├── mpjpe_over_frames.png
│   ├── per_finger_boxplot.png
│   ├── error_scatter.png
│   ├── summary_table.png
│   ├── prediction_grid.png
│   └── sample_overlays/
│       └── *.jpg
├── figures_hamer/                  # HaMeR figures
│   ├── metric_bar.png
│   ├── pa_mpjpe_histogram.png
│   ├── mpjpe_over_frames.png
│   ├── per_finger_boxplot.png
│   ├── error_scatter.png
│   ├── summary_table.png
│   ├── prediction_grid.png
│   └── sample_overlays/
│       └── *.jpg
```

## Scripts

| Script | Description |
|--------|-------------|
| `run_eval_wilor.py` | Run WiLoR evaluation (detect or crop-regress mode) |
| `run_eval_hamer.py` | Run HaMeR evaluation (crop-regress mode) |
| `visualize_results.py` | Generate aggregate analysis plots from results JSON |
| `visualize_predictions.py` | Overlay GT vs predicted keypoints on images (`--model wilor\|hamer`) |
