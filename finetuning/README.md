# WiLoR Finetuning on POV-Surgery

Finetunes the off-the-shelf WiLoR hand-pose estimator on the POV-Surgery
synthetic surgical dataset. Reproduces / matches the finetuned-WiLoR baseline
from the CPCI paper (2293, MICCAI).

## Prerequisites

- GPU with ≥24 GB VRAM (trained on a single RTX PRO 6000 Blackwell Max-Q, 98 GB)
- `aim2` conda env (see `hamer/setup_env.sh` for base setup)
- WiLoR source checkout at `../../WiLoR/` (sibling of `AIM_2_Project/`) with
  `pretrained_models/wilor_final.ckpt` fetched, OR the `wilor_mini` pip
  package installed (the script auto-detects the checkpoint location).

## Required downloads

### 1. POV-Surgery training data (required)

Download `POV_Surgery_data.zip` (61 GB) from the
[POV-Surgery Google Drive](https://drive.google.com/drive/folders/1nSDig2cEHscCPgG10-VcSW3Q1zKge4tP)
and unzip to `../pov_surgery_data/`:

```
pov_surgery_data/
├── demo_data/
│   └── POV_Surgery_data/
│       ├── annotation/       # per-frame MANO + camera pkl files
│       ├── color/            # 1920×1080 JPEGs
│       └── handoccnet_train/
│           ├── 2d_repro_ho3d_style_hocc_cleaned.pkl    # train split
│           └── 2d_repro_ho3d_style_test_cleaned.pkl    # test split
└── data/
    └── bodymodel/
        └── MANO_RIGHT.pkl
```

### 2. MANO model (required)

Download `MANO_RIGHT.pkl` from https://mano.is.tue.mpg.de and place at
`../pov_surgery_data/data/bodymodel/MANO_RIGHT.pkl`.

### 3. WiLoR pretrained weights

No manual step if `wilor_mini` is pip-installed — `train.py` finds the
checkpoint automatically. Otherwise, fetch `wilor_final.ckpt` into
`../../WiLoR/pretrained_models/`.

## Training

```bash
source /mnt/ssd/yuchang/miniforge3/bin/activate aim2
cd AIM_2_Project/finetuning

python train.py \
    --data-dir ../pov_surgery_data \
    --config configs/finetune_pov_surgery.yaml \
    --output-dir ./output \
    --devices 1 \
    --precision 16-mixed
```

Defaults:
- LR 5e-6, AdamW, weight decay 1e-4
- Batch size 16, 50 000 steps (~20 epochs)
- fp16-mixed, gradient clipping 1.0
- Validation every 500 steps, `save_top_k=3` on `val/loss`

Wall-clock on a single RTX PRO 6000 Blackwell Max-Q: ~13.2 hours.

Checkpoints land in `output/checkpoints/`:
- `last.ckpt` — final step, always the one referenced by OOD evals
- `wilor_ft_step=0XXXX_val=0.YYYY.ckpt` — best 3 by `val/loss`
- `wilor_ft_step=0XXXX_val/` — empty stubs left by Lightning's top-k pruning;
  safe to ignore

## Evaluating a finetuned checkpoint on POV-Surgery

Use the standard evaluation script from `eval/`:

```bash
cd AIM_2_Project/eval

python run_eval_wilor.py --mode crop \
    --split full \
    --data-dir ../pov_surgery_data \
    --ckpt-path ../finetuning/output/checkpoints/last.ckpt \
    --output-dir results/finetuned_last
```

Expected final metrics on the POV-Surgery full test set (26 418 frames) at
step 50 k:
- MPJPE 10.59 mm | PA-MPJPE 4.36 mm | PVE 10.11 mm | PA-PVE 4.19 mm
- P2D 29.40 px

Matches the CPCI paper's reported finetuned WiLoR baseline on PA-MPJPE
(paper: 4.33) and PA-PVE (paper: 4.20).

## Files

| File | Purpose |
|---|---|
| `train.py` | Main Lightning entry point. Loads config, creates datasets + `WiLoRFinetune`, logs to TensorBoard. |
| `eval_finetuned.py` | Standalone eval script (superseded by `eval/run_eval_wilor.py --ckpt-path`) |
| `configs/finetune_pov_surgery.yaml` | Hyperparameters + loss weights |
| `datasets/pov_surgery_dataset.py` | PyTorch Dataset with batch format matching WiLoR's `compute_loss` expectations. Precomputes + caches GT. |
| `models/wilor_finetune.py` | `WiLoRFinetune` subclass: single-optimizer training step, vertex L1 loss added. |
| `utils/gt_processing.py` | Shared GT utilities: MANO forward, GRAB→Camera transforms, metric helpers. |

## Non-obvious bits

- **Manual optimization** (`self.automatic_optimization = False` in the
  training step) — because we swap in a simpler single-optimizer setup vs.
  WiLoR's original discriminator-based training. Gradient clipping is
  therefore done manually inside `training_step`, not via `Trainer(gradient_clip_val=)`.
- **`init_renderer=False`** in `WiLoRFinetune(...)` skips the PyTorch3D
  renderer to avoid the TensorBoard mesh-visualization dependency.
- **`ADVERSARIAL: 0.0`** in the config disables WiLoR's discriminator branch
  (we have no mocap data for finetuning).
- **Vertex L1 loss** is added on top of WiLoR's original losses, following
  the CPCI paper's recipe ("L2 on MANO params + L2 on vertex distances").
  Weight defaults to 0.0 in the config — turn on via `VERTICES_3D: 0.05` if
  you want to reproduce that recipe exactly.
