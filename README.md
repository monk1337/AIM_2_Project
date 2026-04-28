# Gloved Hands, Blind Models: Bridging the Domain Gap in Surgical Hand Pose Estimation

BMI 702 / AIM II: Artificial Intelligence in Medicine II
Harvard Department of Biomedical Informatics, Spring 2026

Team: Ankit Pal, Yuchang Su
Instructor: Marinka Zitnik, PhD

This repository contains the full source for the four-model benchmark, the
fine-tuning ablations, the inference-time ensemble, the multi-teacher
distillation, and the mesh-shape preservation regularizer described in the
final report.

## What is in this repository

```
.
├── README.md           this file
├── SETUP.md            install + dataset + checkpoint download
├── requirements.txt    Python deps (Python 3.10, CUDA 12.1, PyTorch 2.3)
├── Makefile            entry points for every experiment
├── src/
│   ├── eval/           four-model benchmark, ensemble, per-seq, per-finger
│   ├── train/          ft_wilor, ft_wilor_mixed (Exp B), ft_wilor_anchored
│   │                   ft_wilor_distill (Exp I), ft_honet, precompute_teacher
│   ├── viz/            mesh + keypoint overlays, export to JSON for the dashboard
│   └── dev/            debugging scripts, kept for reference
├── scripts/
│   ├── download_pov.sh
│   ├── download_aria.sh
│   ├── download_checkpoints.sh
│   └── data_pipeline/  midterm-era data-collection helpers
├── external/           pinned upstream model repos (submodules) + patches
│   ├── README.md       set-up commands and pinned commit table
│   └── PATCHES/
│       └── hamer.patch (50-line modification to upstream HaMeR)
├── midterm/            archived midterm content (kept for reference)
│   ├── README.md       original midterm README
│   ├── PHASE1_EVAL_REPORT.md
│   ├── PHASE2_FT_REPORT.md
│   ├── eval/           midterm WiLoR/HaMeR/HandOccNet eval scripts
│   └── hamer/          midterm HaMeR-specific scripts
├── data/               (empty; populated by scripts/download_*.sh)
├── checkpoints/        (empty; populated by scripts/download_checkpoints.sh)
└── results/            (empty; populated by `make eval-* / make ft-*`)
```

## Quick start

1. Read `SETUP.md` and follow it once.
2. Smoke test:
   ```bash
   make eval-wilor-pov-quick
   ```
   Expect PA-MPJPE around 11 mm if everything is wired correctly.
3. Reproduce the headline:
   ```bash
   make ft-wilor-mixed         # Exp B
   make eval-wilor-aria        # Aria HSAM PA-MPJPE
   ```

## Reproducibility notes

* Every experiment in the report has a `Makefile` target. The targets are
  thin wrappers around `python -m src.<module>` invocations; passing extra
  arguments to the underlying script works as expected.
* Custom fine-tuning checkpoints are not redistributed. Re-run training to
  reproduce; roughly 30 hours of A100 time covers the full ablation matrix.
* Off-the-shelf checkpoints are available from each model's upstream repo
  (see `external/README.md`); `scripts/download_checkpoints.sh` automates
  most of it.
* The Aria-val canonical filter sidecars (`reject_keys_all_20260419.json`,
  `mps_v2_val_20260419.json`) are mirrored in `aaditya/phase0-artifacts` on
  HuggingFace and are required for the n=2,333 v2-clean evaluation set.

## Headline numbers (from the final report)

| Method                                | POV-Surgery PA | Aria HSAM PA |
| ------------------------------------- | -------------- | ------------ |
| WiLoR off-shelf                       | 10.66 mm       | 14.91 mm     |
| WiLoR FT mixed (Exp B)                | 4.30 mm        | 9.00 mm      |
| WiLoR FT anchored (Exp B+, lambda=0.5) | 5.71 mm        | 14.39 mm     |

| Method (Aria-MPS gold)                | Aria-MPS PA    |
| ------------------------------------- | -------------- |
| Mesh Graphormer off-shelf             | 12.00 mm       |
| HandOccNet off-shelf                  | 13.88 mm       |
| Ensemble (0.75 MGFM + 0.25 HONet)     | 11.52 mm       |
| WiLoR distilled from ensemble (Exp I) | 27.87 mm       |

## Citations

If you use this code, please cite the report and the upstream models. See
`midterm/PHASE1_EVAL_REPORT.md` and `midterm/PHASE2_FT_REPORT.md` for the
midterm-stage analysis.
