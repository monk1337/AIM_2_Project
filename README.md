# Gloved Hands, Blind Models: Bridging the Domain Gap in Surgical Hand Pose Estimation

**BMI 702 / AIM II: Artificial Intelligence in Medicine II**
Harvard Department of Biomedical Informatics, Spring 2026

**Team:** Ankit Pal, Yuchang Su
**Instructor:** Marinka Zitnik, PhD

This repository contains the full source for the four-model benchmark, the
fine-tuning ablations, the inference-time ensemble, the multi-teacher
distillation, and the mesh-shape preservation regularizer described in the
final report.

## TL;DR

We benchmark four hand-pose models (WiLoR, HaMeR, HandOccNet, Mesh Graphormer)
on synthetic surgical data (POV-Surgery) and on real OR egocentric data
(Aria-Surgical-Hand-Pose, with two parallel ground-truth sources). The headline
finding is the **pre-training inversion**: pre-training data, not architecture,
decides which surgical hands a model can see. Mixed fine-tuning of WiLoR cuts
real-OR error by 40 %, and a multi-teacher ensemble + distillation pipeline
delivers the first single-model sub-30 mm result on Aria-MPS gold.

## Repository layout

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
│   ├── viz/            mesh + keypoint overlays, dashboard exports
│   └── dev/            debugging scripts kept for reference
├── scripts/
│   ├── download_pov.sh           public POV-Surgery (project page link)
│   ├── download_aria.sh          private Aria (placeholder; see below)
│   ├── download_checkpoints.sh   off-shelf model weights
│   └── data_pipeline/            midterm data-collection helpers
├── external/           pinned upstream model repos (as submodules) + patches
│   ├── README.md       set-up commands and pinned commit SHAs
│   └── PATCHES/
│       └── hamer.patch (50-line modification to upstream HaMeR)
├── midterm/            archived midterm content (kept for reference)
├── data/               empty; populated by scripts/download_*.sh
├── checkpoints/        empty; populated by scripts/download_checkpoints.sh
└── results/            empty; populated by `make eval-* / make ft-*`
```

## Datasets used in this study

| Dataset | Domain | n | Source | Access |
|---|---|---|---|---|
| POV-Surgery (test split) | Synthetic egocentric surgery | 28,802 | https://batfacewayne.github.io/POV_Surgery_io/ | Public |
| Aria-Surgical-Hand-Pose (val v2-clean) | Real-OR egocentric | 2,333 | Beth Israel Hospital (collected by the authors) | Private; available through a private HuggingFace folder by request |

## Quick start

```bash
# 1. Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. External model repos (one-time, ~800 MB total)
cd external && bash setup.sh && cd ..

# 3. Datasets
bash scripts/download_pov.sh data/pov_surgery     # public, see SETUP.md
bash scripts/download_aria.sh data/aria_val       # private, request access
bash scripts/download_checkpoints.sh checkpoints  # off-shelf weights

# 4. Smoke test (should print PA-MPJPE ~11 mm)
make eval-wilor-pov

# 5. Reproduce the headline (Exp B, ~3 h on a single A100)
make ft-wilor-mixed
make eval-wilor-aria
```

## Path conventions and overrides

The code was originally authored for a single training environment whose
filesystem was rooted at `/workspace/`. Every script reads its data, checkpoint,
and output paths through one of two mechanisms:

* `Makefile` targets accept `DATA=`, `CKPT=`, and `RESULTS=` overrides:
  ```bash
  make eval-wilor-pov DATA=/mnt/datasets CKPT=/mnt/ckpt RESULTS=/mnt/out
  ```
* The Python scripts under `src/` accept the same paths via `--data`,
  `--ckpt`, `--out`, and similar flags. Defaults are visible at the top of
  each file as the `argparse` `default=` argument; edit those if you cannot
  use the Makefile.

If you see a hardcoded `/workspace/...` default in a script, it is the
training-environment default. Override it with one of the two mechanisms
above. No script writes outside the path you provide.

## Headline numbers (from the final report)

PA-MPJPE in mm; lower is better.

| Method                                  | POV-Surgery | Aria HSAM | Aria-MPS gold |
|-----------------------------------------|-------------|-----------|---------------|
| WiLoR off-shelf                         | 10.66       | 14.91     | 41.74         |
| Mesh Graphormer off-shelf               | 41.92       | 40.98     | 12.00         |
| HandOccNet off-shelf                    | 39.09       | 40.43     | 13.88         |
| **WiLoR FT mixed (Exp B)**              | **4.30**    | **9.00**  | 41.67         |
| WiLoR FT anchored (Exp B+, lambda=0.5)  | 5.71        | 14.39     | 41.82         |
| **Inference ensemble (0.75 MGFM + 0.25 HONet)** | n/a | n/a    | **11.52**     |
| **WiLoR distilled from ensemble (Exp I)**       | 6.40 | 35.64 | **27.87**     |

The final report carries the full ablation table, the per-finger view, and the
trade-off discussion.

## Reproducibility notes

* **Custom fine-tuning checkpoints are not redistributed.** Re-run training
  with the `make ft-*` targets to reproduce; roughly 30 hours of A100 time
  covers the full ablation matrix (Exp A, B, B+, C, D, E, F, I).
* **Off-the-shelf checkpoints** are available from each model's upstream repo
  (see `external/README.md`); `scripts/download_checkpoints.sh` automates the
  steps that can be automated and prints instructions for the rest.
* **POV-Surgery** is publicly released by Wang et al. (2023) at
  https://batfacewayne.github.io/POV_Surgery_io/. Follow the project page to
  obtain the dataset.
* **Aria-Surgical-Hand-Pose** is a privately-collected dataset from
  Beth Israel Hospital. It is not publicly distributed; request access from
  the authors. The `scripts/download_aria.sh` script is a placeholder that
  documents the expected on-disk layout for evaluation to run.

## Citing this work

If you use this code, please cite the final report. The report's bibliography
contains the full list of model and dataset citations
(WiLoR, HaMeR, HandOccNet, Mesh Graphormer, POV-Surgery, MANO).

The midterm-stage analysis is preserved under `midterm/PHASE1_EVAL_REPORT.md`
and `midterm/PHASE2_FT_REPORT.md`.

## License

Code is released under the MIT license (see LICENSE if present, otherwise
contact the authors). The forked upstream model repositories under
`external/` retain their own licenses; consult each repo's LICENSE before
redistribution.
