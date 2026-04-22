# WiLoR Finetuning: In-Domain + OOD Evaluation

Comparison of off-the-shelf WiLoR against POV-Surgery-finetuned WiLoR across three datasets:
one in-domain (synthetic surgical), one out-of-distribution real-world with pseudo-labels
(Aria smart-glasses validation), and one qualitative real-world surgical video (AIxSuture).

## TL;DR

| Dataset | Type | Finetuning outcome |
|---|---|---|
| POV-Surgery (synthetic) | in-domain | Massive gains on 3D metrics (−79% MPJPE, −59% PA-MPJPE). P2D degrades slightly. |
| Aria validation (real) | OOD, pseudo-GT | PA-MPJPE nearly identical (−2%). P2D regresses by 81%, likely due to fish-eye camera differing from POV-Surgery pinhole. |
| AIxSuture (real surgery) | OOD, qualitative | No GT. Both models produce hand-like skeletons on visible gloved hands, with finetuned slightly more tied to POV-Surgery visual priors. |

**Bottom line:** finetuning dramatically improves POV-Surgery 3D accuracy but does not transfer cleanly to other real-world camera setups. Shape priors (PA-MPJPE) generalize reasonably; absolute 2D alignment (P2D) is sensitive to camera intrinsics and image statistics the model was trained on.

## Finetuning Setup (Recap)

- Base model: `wilor_final.ckpt` from `wilor_mini`
- Training data: POV-Surgery official train split (~38k samples), right hand only
- Loss: WiLoR's original 3D-joint L1 + 2D-joint L1 + MANO-param MSE, following the CPCI paper recipe
- Optimizer: AdamW, LR=5e-6, weight decay=1e-4
- Trainer: PyTorch Lightning, single RTX PRO 6000 Blackwell Max-Q (98 GB), fp16-mixed, 50k steps (~20 epochs), batch size 16
- Wall-clock: ~13.2 h
- Final checkpoint used: `AIM_2_Project/finetuning/output/checkpoints/last.ckpt` (step 50k)

## Main Comparison Table

Metric conventions: all errors in millimetres for 3D, pixels for 2D.
Cells marked "—" indicate the metric isn't trustworthy or GT isn't available for that dataset.

### POV-Surgery (synthetic, full test set — 26 418 frames, 17 sequences)

| Metric | Off-the-shelf | Finetuned (step 50k) | Δ (rel) | CPCI paper ft-WiLoR ref |
|---|---:|---:|---:|---:|
| MPJPE (mm)    | 50.36 | **10.59** | −79.0% | 13.72 |
| PA-MPJPE (mm) | 10.69 | **4.36**  | −59.2% | 4.33  |
| PVE (mm)      | 47.92 | **10.11** | −78.9% | 12.91 |
| PA-PVE (mm)   | 10.01 | **4.19**  | −58.1% | 4.20  |
| P2D (px)      | **25.99** | 29.40 | +13.1% | 18.48 |

Note: our finetuned MPJPE/PVE beats the CPCI paper's reported finetuned WiLoR;
PA-MPJPE/PA-PVE are on par. P2D is slightly worse than the CPCI reference,
likely explained by our (simpler) L1 2D-loss weighting not matching the paper's recipe.

### Aria validation (real-world pseudo-GT — 784 hand samples across PR83 + PR84)

Aria's `eval_joints_3d` is not in the same camera frame used for 2D projection,
so only frame-invariant (PA) 3D metrics are reported; `vertices_3d` isn't provided.
GT 2D joints are pseudo-labels (Guillaume, MediaPipe-style). Left hands (346/784)
were evaluated by horizontal image flip + prediction un-flip.

| Metric | Off-the-shelf | Finetuned (step 50k) | Δ (rel) |
|---|---:|---:|---:|
| PA-MPJPE (mm) | 42.08 | **41.21** | −2.1% |
| P2D (px)      | **15.39** | 27.84 | +80.9% |

`evaluated = 784/784` samples for both runs. No detection failures.

Per-finger PA-MPJPE (finetuned, mm): see `results/aria_finetuned/wilor_ood_results.json`.

**Interpretation**
- PA-MPJPE barely changes: hand-shape priors generalize. Both off-the-shelf and
  finetuned produce similarly deformed hand meshes once scale/rotation/translation
  are factored out.
- P2D doubles. Two likely causes:
  1. **Camera mismatch.** Aria RGB is a fish-eye approximated by a pinhole
     (`K = [[606.5, 0, 704], [0, 606.5, 704], [0, 0, 1]]`), whereas POV-Surgery
     training data has `fx=fy=1198.44` with `cy=175.2`. Finetuning specialized
     the model to POV-Surgery's principal point / FOV geometry.
  2. **Pseudo-label noise.** Aria `joints_2d` are Guillaume-authored detections,
     not geometric ground truth, so absolute P2D values are noisier than POV-Surgery's.

### AIxSuture (real-world surgical video — qualitative only, 30 frames from one video)

No hand-pose ground truth in this dataset (OSATS skill scores only), so no
quantitative metrics. Used here purely to inspect model behaviour on real
surgical footage.

| Metric | Off-the-shelf | Finetuned |
|---|---:|---:|
| Evaluated | 30 detected / 30 total | 30 / 30 |
| All metrics | — (no GT) | — (no GT) |

Source: `Package 11.zip` from Zenodo record 7940583 (AIxSuture), video `Z49G.mp4`
(5:06, 1920×1080, 30 fps), 30 frames sampled at 1 Hz.

**Qualitative observation**: both models produce plausible 21-joint hand
skeletons on frames where a blue-gloved hand is clearly visible. On frames
where the hand is off-centre, partially occluded, or the placeholder center-crop
bbox doesn't cover it, both models hallucinate a skeleton; visually the
finetuned predictions appear slightly more "synthetic-POV-Surgery-like".
See grids in `results/aixsuture_raw/figures/prediction_grid.png` and
`results/aixsuture_finetuned/figures/prediction_grid.png`.

## Visualization Artifacts

Per-model overlays under `ood_eval/results/<dataset>_<variant>/figures/`:

```
results/aria_raw/figures/
  sample_overlays/*.jpg        (per-sample 3-panel overlays, PA-MPJPE-sorted)
  prediction_grid.png          (composite, 4 categories: best/median/worst/spread)
results/aria_finetuned/figures/
  sample_overlays/*.jpg
  prediction_grid.png
results/aixsuture_raw/figures/
  sample_overlays/*.jpg        (evenly-spread single-panel overlays)
  prediction_grid.png
results/aixsuture_finetuned/figures/
  sample_overlays/*.jpg
  prediction_grid.png
```

Per-sample JPEGs tagged `best/median/worst/spread` (Aria) or `spread`
(AIxSuture, no-GT). Left-hand samples carry a `[LEFT]` badge.

### Same-frame raw-vs-finetuned comparison

Produced by `compare_raw_vs_ft.py` — picks the same frames for both models so
predictions can be compared directly.

- **Aria** (`results/compare_aria/compare_grid.png`): 4 frames where finetuning
  improved **P2D** the most + 4 frames where finetuning regressed the most.
  Ranking uses P2D (pixel error) rather than PA-MPJPE because the 2D
  visualization would otherwise disagree with the metric — on Aria, the
  3D GT (`eval_joints_3d`) is not in the projection camera frame, so 3D
  PA-MPJPE can "win" for a prediction whose 2D projection is clearly off.
  P2D deltas on Aria: best finetuning improvement +23.3 px, worst
  regression −41.6 px.
- **AIxSuture** (`results/compare_aixsuture/compare_grid.png`): 8 evenly-spread
  frames shown side-by-side — qualitative only, no GT.

Per-frame side-by-side JPEGs land in `results/compare_<dataset>/sample_overlays/`.

## Reproducing

```bash
source /mnt/ssd/yuchang/miniforge3/bin/activate aim2
cd /mnt/ssd/yuchang/SurgicalVLA/AIM_2_Project/ood_eval

# 1. Build common-format datasets (one-time)
python adapters/aria_to_common.py          # ~5 min, writes 784 samples to common_format/aria/
python adapters/aixsuture_to_common.py     # downloads 4.4 GB, samples 30 frames, cleans up

# 2. Eval on Aria (raw + finetuned)
python run_eval_ood.py --dataset-dir common_format/aria \
    --output-dir results/aria_raw --device cuda
python run_eval_ood.py --dataset-dir common_format/aria \
    --output-dir results/aria_finetuned --device cuda \
    --ckpt-path ../finetuning/output/checkpoints/last.ckpt

# 3. Eval on AIxSuture (raw + finetuned, predictions only)
python run_eval_ood.py --dataset-dir common_format/aixsuture \
    --output-dir results/aixsuture_raw --device cuda
python run_eval_ood.py --dataset-dir common_format/aixsuture \
    --output-dir results/aixsuture_finetuned --device cuda \
    --ckpt-path ../finetuning/output/checkpoints/last.ckpt

# 4. Per-model visualizations
for DS in aria_raw aria_finetuned aixsuture_raw aixsuture_finetuned; do
    python visualize_ood.py \
        --results results/${DS}/wilor_ood_results.json \
        --dataset-dir common_format/${DS%_*} \
        --output-dir results/${DS}/figures/sample_overlays --n-each 2
done

# 5. Raw-vs-finetuned side-by-side on the same frames (ranked by P2D by default)
python compare_raw_vs_ft.py \
    --raw results/aria_raw/wilor_ood_results.json \
    --ft  results/aria_finetuned/wilor_ood_results.json \
    --dataset-dir common_format/aria \
    --output-dir results/compare_aria \
    --n-wins 4 --n-losses 4 --rank-by p2d

python compare_raw_vs_ft.py \
    --raw results/aixsuture_raw/wilor_ood_results.json \
    --ft  results/aixsuture_finetuned/wilor_ood_results.json \
    --dataset-dir common_format/aixsuture \
    --output-dir results/compare_aixsuture \
    --n-spread 8
```

## Pipeline Architecture

```
AIM_2_Project/ood_eval/
├── README.md                           # (this file's location parent; not generated)
├── RESULTS.md                          # this file
├── adapters/
│   ├── aria_to_common.py               # HuggingFace arrow -> common_format/aria/
│   └── aixsuture_to_common.py          # Zenodo Package 11 -> common_format/aixsuture/
├── common_format/
│   ├── aria/                           # 481 rotated PNGs + 784-sample pkl + 6 sanity overlays
│   │   ├── images/PR{83,84}_*.png
│   │   ├── samples.pkl
│   │   └── sanity_overlays/*.png
│   └── aixsuture/                      # 30 PNGs + 30-sample pkl (no GT)
│       ├── images/Z49G_frame_*.png
│       └── samples.pkl
├── run_eval_ood.py                     # WiLoR runner; CLI: --dataset-dir / --ckpt-path
├── visualize_ood.py                    # reads wilor_ood_results.json, emits overlays
└── results/
    ├── aria_raw/{wilor_ood_results.json, figures/}
    ├── aria_finetuned/...
    ├── aixsuture_raw/...
    └── aixsuture_finetuned/...
```

Common-format sample schema (`samples.pkl` → `list[dict]`):
```python
{
    "frame_id":    "PR83/000123/right",      # unique key
    "image_path":  "images/PR83_000123.png", # relative to samples.pkl
    "K":           np.float32 (3, 3),        # per-sample intrinsics
    "joints_3d":   np.float32 (21, 3) | None,
    "joints_3d_frame_trustworthy": bool,     # gates absolute MPJPE/PVE
    "vertices_3d": np.float32 (778, 3) | None,
    "joints_2d":   np.float32 (21, 2) | None,# pixels, OpenPose order
    "bbox":        np.float32 (4,),          # xyxy, hand region
    "is_right":    bool,                     # runner flips images for False
    "hand_side":   "left" | "right",
}
```

`run_eval_ood.py` reads this, runs WiLoR's `predict_with_bboxes` (crop-regress),
flips+un-flips for left hands, and writes a JSON with both metrics and raw
predictions for downstream visualization — no re-inference needed for viz.

## Known Caveats

1. **Aria `eval_joints_3d`** is not geometrically consistent with `joints_2d`
   (PnP residual ~50 px). Do not use non-PA 3D metrics on Aria.
2. **Aria `joints_2d`** are pseudo-labels with unknown noise floor.
3. **Aria fish-eye distortion**: the provided pinhole K is an approximation;
   peripheral hands will have higher apparent P2D.
4. **AIxSuture bbox placeholder**: we use a static 60% center crop as a weak
   hand-location prior; frames where the hand is off-centre will have the
   model regress into a region with no hand, producing spurious skeletons.
5. **Handedness flip**: we mirror left-hand images to run as right hands, then
   un-flip predictions. This assumes WiLoR's MANO decoder is symmetric under
   image mirroring, which is approximately true but not exact.
