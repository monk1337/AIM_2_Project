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
(Aria-Surgical-Hand-Pose, with two parallel ground-truth sources). The
headline finding is the **pre-training inversion**: pre-training data, not
architecture, decides which surgical hands a model can see. Mixed fine-tuning
of WiLoR cuts real-OR error by 40%, and a multi-teacher ensemble plus
distillation pipeline delivers the first single-model sub-30 mm result on
Aria-MPS gold.

## Headline numbers (PA-MPJPE, mm; lower is better)

| Method                                          | POV-Surgery | Aria HSAM | Aria-MPS gold |
|-------------------------------------------------|-------------|-----------|---------------|
| WiLoR off-shelf                                 | 10.66       | 14.91     | 41.74         |
| HaMeR off-shelf                                 | 11.23       | 15.63     | 41.64         |
| HandOccNet off-shelf                            | 39.09       | 40.43     | 13.88         |
| Mesh Graphormer off-shelf                       | 41.92       | 40.98     | 12.00         |
| **WiLoR FT mixed (Exp B)**                      | **4.30**    | **9.00**  | 41.67         |
| WiLoR FT anchored (Exp B+, lambda_mesh = 0.5)   | 5.71        | 14.39     | 41.82         |
| **Inference ensemble (0.75 MGFM + 0.25 HONet)** | n/a         | n/a       | **11.52**     |
| **WiLoR distilled from ensemble (Exp I)**       | 6.40        | 35.64     | **27.87**     |

## Repository layout

```
.
├── README.md           this file
├── SETUP.md            install, datasets, checkpoints, external repos
├── requirements.txt    Python deps (Python 3.10, CUDA 12.1, PyTorch 2.3)
├── Makefile            one entry point per experiment (eval-*, ft-*, viz-*)
├── src/
│   ├── eval/           four-model benchmark + ensemble + per-seq + per-finger
│   ├── train/          ft_wilor (Exp A), ft_wilor_mixed (Exp B),
│   │                   ft_wilor_anchored (Exp B+), ft_wilor_distill (Exp I),
│   │                   ft_honet (Exp E/F), precompute_teacher
│   ├── viz/            mesh + keypoint overlays, dashboard exporters
│   └── dev/            small debug scripts kept for reference
├── scripts/
│   ├── download_pov.sh              public POV-Surgery (project page link)
│   ├── download_aria.sh             private Aria placeholder + layout
│   ├── download_checkpoints.sh      off-shelf model weights
│   └── data_pipeline/               midterm data-collection helpers
├── external/                        pinned upstream model repos + patches
│   ├── README.md                    pinned commit SHAs and setup commands
│   └── PATCHES/hamer.patch          50-line modification to upstream HaMeR
├── midterm/                         archived midterm content
│   ├── README.md
│   ├── PHASE1_EVAL_REPORT.md
│   ├── PHASE2_FT_REPORT.md
│   ├── eval/                        midterm-era eval scripts
│   └── hamer/                       midterm HaMeR-specific scripts + viz
├── reports/                         midterm PDFs
├── data/, checkpoints/, results/    empty placeholders (populated at runtime)
```

## Hardware requirements

| Stage                  | Minimum         | Recommended            |
|------------------------|-----------------|------------------------|
| Off-shelf evaluation   | 16 GB GPU       | A100 / A6000           |
| Fine-tuning (Exp A/B)  | 24 GB GPU       | A100 (single)          |
| Distillation (Exp I)   | 40 GB GPU       | A100 80 GB             |
| Disk (datasets)        | 25 GB free      | 30 GB                  |
| Disk (off-shelf ckpt)  | 16 GB free      |                        |
| Disk (custom FT ckpt)  | 115 GB free     | (full ablation matrix) |

CPU-only paths exist for the eval scripts but are 50-100x slower; not recommended.

## Datasets used in this study

| Dataset                                         | Domain                  | n      | Source                                            | Access                                                            |
|-------------------------------------------------|-------------------------|--------|---------------------------------------------------|-------------------------------------------------------------------|
| POV-Surgery (test split)                        | Synthetic ego. surgery  | 28,802 | https://batfacewayne.github.io/POV_Surgery_io/    | Public                                                            |
| Aria-Surgical-Hand-Pose (val v2-clean)          | Real-OR egocentric      | 2,333  | Beth Israel Hospital (collected by the authors)   | Private; available through a private HuggingFace folder by request |

## Quick start

```bash
# 1. Python environment
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. External model repos (one-time, ~800 MB total)
#    Follows the script in external/README.md to pin commits and apply hamer.patch.
( cd external && bash setup.sh )    # if you create one; otherwise read external/README.md

# 3. Datasets
bash scripts/download_pov.sh         data/pov_surgery   # public; see project page
bash scripts/download_aria.sh        data/aria_val      # private; request access

# 4. MANO model files (registration required, ~258 MB)
#    Download MANO_RIGHT.pkl + MANO_LEFT.pkl from https://mano.is.tue.mpg.de/
#    and place them under data/mano/

# 5. Off-the-shelf checkpoints (~16 GB)
bash scripts/download_checkpoints.sh checkpoints

# 6. Smoke test (should print PA-MPJPE around 11 mm)
make eval-wilor-pov DATA=data CKPT=checkpoints RESULTS=results

# 7. Reproduce the headline (Exp B, ~3 h on a single A100)
make ft-wilor-mixed
make eval-wilor-aria
```

## Reproducing each result in the report

Each row in the headline table maps to one or two `Makefile` targets.

| Result                                          | Targets                                               | Wall-clock on A100 |
|-------------------------------------------------|-------------------------------------------------------|--------------------|
| Off-shelf benchmark (Tab. 2)                    | `eval-{wilor,hamer,handoccnet,mgfm}-{pov,aria}`        | ~30 min total      |
| Per-sequence Aria breakdown                     | `eval-per-seq`                                         | ~10 min            |
| Per-finger PA-MPJPE                             | `eval-per-finger`                                      | ~5 min             |
| Exp A (POV-only FT)                             | `ft-wilor-pov` then `eval-wilor-{pov,aria}`            | ~3 h               |
| **Exp B (mixed FT, headline)**                  | `ft-wilor-mixed` then `eval-wilor-{pov,aria}`          | ~3 h               |
| Exp B+ (anchored regularizer)                   | `ft-wilor-anchored`                                    | ~3 h               |
| Exp E/F (HandOccNet FT, expected to regress)    | `ft-honet`                                             | ~2 h               |
| Inference-time ensemble                         | `eval-ensemble`                                        | ~20 min            |
| **Exp I (multi-teacher distillation)**          | `precompute-teacher` then `ft-wilor-distill`           | ~6 h               |

`make help` lists every target.

## Path conventions and overrides

The code was originally authored for a single training environment whose
filesystem was rooted at `/workspace/`. **Every script reads its data,
checkpoint, and output paths through one of three mechanisms:**

1. `Makefile` accepts overrides on the command line:
   ```bash
   make eval-wilor-pov DATA=/mnt/datasets CKPT=/mnt/ckpt RESULTS=/mnt/out
   ```
2. The Python scripts under `src/` accept the same paths via `--data`,
   `--ckpt`, `--out`, and similar flags (visible at the top of each file's
   `argparse` section).
3. Two environment variables override the Aria-canonical filter sidecar:
   ```bash
   export AIM2_SIDECAR_DIR=/path/to/phase0_sidecars   # default: /workspace/datasets/phase0_sidecars
   export AIM2_ARIA_VAL_DIR=/path/to/aria_val/data    # default: /workspace/datasets/aria_val/data
   export AIM2_POV_DIR=/path/to/pov_surgery/data      # default: /workspace/datasets/pov_surgery/data
   export AIM2_CACHE_DIR=/path/to/cache               # ensemble teacher cache
   ```

If a script defaults to a `/workspace/...` path, that is the original training
environment default; the path overrides above redirect to your filesystem
without any code change.

## How to read the codebase

* **Entry points:** `Makefile` and `python -m src.<module>` invocations.
* **Shared utilities:**
  * `src/eval/eval_loader.py` - dataset loaders, bbox derivation, the v2-clean filter
  * `src/eval/eval_metrics.py` - PA-MPJPE, PVE, Procrustes alignment
  * `src/eval/eval_joint_orders.py` - MANO / OpenPose / Aria-MPS canonical permutations
* **One model per file** under `src/eval/eval_{wilor,hamer,handoccnet,meshgraphormer}.py`. They all follow the same crop-regress contract: load data, run model, permute to OP-21, compute metrics, dump JSON.
* **One recipe per file** under `src/train/ft_wilor*.py` and `src/train/ft_honet.py`. Each is self-contained; they share dataset loaders via `src/eval/eval_loader.py`.
* **Visualization** under `src/viz/` mirrors the eval shape; `viz_seq_mesh.py` is the canonical mesh-overlay renderer and feeds the dashboard.

## Troubleshooting

| Symptom                                                            | Likely cause                                                                                                                        | Fix                                                                                                                                |
|--------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `ModuleNotFoundError: wilor_mini`                                  | wilor_mini not installed                                                                                                            | `pip install wilor_mini`                                                                                                           |
| `ModuleNotFoundError: eval_loader` (etc.)                          | running a script directly without the parent path                                                                                   | invoke via `python -m src.eval.<name>` or via the Makefile                                                                         |
| `FileNotFoundError: .../reject_keys_all_20260419.json`             | sidecar not at default `/workspace/datasets/phase0_sidecars/`                                                                       | `export AIM2_SIDECAR_DIR=/your/path` or pass `--sidecar_dir`                                                                       |
| WiLoR YOLO detector returns 0 hands on POV                         | known issue: green gloves trip the detector                                                                                         | use crop-regress mode (the default in this repo); ground-truth bboxes are supplied automatically                                    |
| HaMeR runs out of RAM on macOS / MPS                               | HaMeR's MANO layer uses float64; not supported on MPS                                                                               | run on CUDA, or fall back to CPU (slow)                                                                                            |
| `OSError: cannot identify image file` on Aria                      | image is in display frame but `joints_2d` is in raw sensor frame                                                                    | always apply CCW90 to `joints_2d` before pairing with the image (the loaders do this automatically)                                |

## Reproducibility notes

* **Custom fine-tuning checkpoints are not redistributed.** Re-run training
  with the `make ft-*` targets to reproduce; roughly 30 GPU-hours of A100
  time covers the full ablation matrix (Exp A, B, B+, C, D, E, F, I).
* **Off-the-shelf checkpoints** are available from each model's upstream
  repo (see `external/README.md`). `scripts/download_checkpoints.sh`
  automates what it can and prints upstream URLs for the rest.
* **POV-Surgery** is publicly released by Wang et al. (2023) at
  https://batfacewayne.github.io/POV_Surgery_io/.
* **Aria-Surgical-Hand-Pose** is a privately collected dataset from
  Beth Israel Hospital. It is not publicly distributed; request access from
  the report authors. The placeholder script
  `scripts/download_aria.sh` documents the expected on-disk layout.

## Citing this work

If you use this code, please cite the final report. The report's
bibliography contains the full list of model and dataset citations
(WiLoR, HaMeR, HandOccNet, Mesh Graphormer, POV-Surgery, MANO, Aria).

The midterm-stage analysis is preserved under
`midterm/PHASE1_EVAL_REPORT.md` and `midterm/PHASE2_FT_REPORT.md`.

## Acknowledgments

This work uses MANO (Romero, Tzionas, Black, 2017), POV-Surgery
(Wang et al., 2023), Aria recordings collected at Beth Israel Hospital, and
four off-shelf models: WiLoR (Potamias et al., CVPR 2025), HaMeR (Pavlakos
et al., CVPR 2024), HandOccNet (Park et al., CVPR 2022), and Mesh Graphormer
(Lin et al., ICCV 2021). We thank the authors of these works for open-sourcing
their code and pre-trained weights.

## License

Code in `src/`, `scripts/`, `Makefile`, and the top-level docs is released
under the MIT license (see `LICENSE` if present, otherwise contact the
authors). The forked upstream model repositories under `external/` retain
their own licenses; consult each repo's LICENSE before redistribution.
