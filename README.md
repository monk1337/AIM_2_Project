# Gloved Hands, Blind Models: Bridging the Domain Gap in Surgical Hand Pose Estimation

**BMI 702 / AIM II, Harvard DBMI, Spring 2026.**
**Authors:** Ankit Pal, Yuchang Su.
**Instructor:** Marinka Zitnik, PhD.

Code to reproduce the four-model surgical hand-pose benchmark, the
fine-tuning ablations, the inference-time ensemble, the multi-teacher
distillation, and the mesh-shape preservation regularizer described in the
final report.

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

## Layout

```
.
├── README.md           this file (everything you need)
├── LICENSE             MIT
├── requirements.txt    Python deps (Python 3.10, CUDA 12.1, PyTorch 2.3)
├── Makefile            one entry point per experiment
├── src/
│   ├── eval/           four-model benchmark + ensemble + per-seq + per-finger
│   ├── train/          ft_wilor* (Exp A/B/B+/I), ft_honet, precompute_teacher
│   └── viz/            mesh + keypoint overlays, dashboard exporters
├── scripts/
│   ├── download_pov.sh           public POV-Surgery (project-page link)
│   ├── download_aria.sh          private Aria placeholder
│   └── download_checkpoints.sh   off-shelf model weights
└── external/
    └── PATCHES/hamer.patch       50-line modification to upstream HaMeR
```

## Hardware

| Stage                  | Minimum     | Recommended   |
|------------------------|-------------|---------------|
| Off-shelf evaluation   | 16 GB GPU   | A100 / A6000  |
| Fine-tuning (Exp A/B)  | 24 GB GPU   | A100 single   |
| Distillation (Exp I)   | 40 GB GPU   | A100 80 GB    |
| Disk (datasets)        | 25 GB free  | 30 GB         |
| Disk (off-shelf ckpt)  | 16 GB free  |               |
| Disk (custom FT ckpt)  | 115 GB free | full ablation |

## Setup

### 1. Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. External model repositories

The four benchmarked models live in their own upstream repos. Pin these
exact commits and apply the HaMeR patch.

| Repo            | Upstream URL                                       | Commit    | Patch                      |
|-----------------|----------------------------------------------------|-----------|----------------------------|
| WiLoR           | https://github.com/rolpotamias/WiLoR.git           | `fcb9113` | (none)                     |
| HandOccNet      | https://github.com/namepllet/HandOccNet.git        | `65ba997` | (none)                     |
| MeshGraphormer  | https://github.com/microsoft/MeshGraphormer.git    | `27f7cdb` | (none)                     |
| hamer           | https://github.com/geopavlakos/hamer.git           | `3a01849` | `external/PATCHES/hamer.patch` |

```bash
cd external
for entry in \
    "WiLoR https://github.com/rolpotamias/WiLoR.git fcb9113" \
    "HandOccNet https://github.com/namepllet/HandOccNet.git 65ba997" \
    "MeshGraphormer https://github.com/microsoft/MeshGraphormer.git 27f7cdb" \
    "hamer https://github.com/geopavlakos/hamer.git 3a01849"; do
  set -- $entry
  git clone "$2" "$1" && ( cd "$1" && git checkout "$3" )
done
( cd hamer && git apply ../PATCHES/hamer.patch )
cd ..
```

### 3. Datasets

| Dataset                                 | Source                                            | Access                                                            |
|-----------------------------------------|---------------------------------------------------|-------------------------------------------------------------------|
| POV-Surgery (test split, n = 28,802)    | https://batfacewayne.github.io/POV_Surgery_io/    | Public                                                            |
| Aria-Surgical-Hand-Pose (val v2-clean, n = 2,333) | Beth Israel Hospital                  | Private; available through a private HuggingFace folder by request |

```bash
bash scripts/download_pov.sh   data/pov_surgery
bash scripts/download_aria.sh  data/aria_val
```

`scripts/download_pov.sh` documents the on-disk layout expected by the
loaders. `scripts/download_aria.sh` is a placeholder; contact the report
authors for access to the private Aria folder.

### 4. MANO model files (registration required, ~258 MB)

Download `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` from
https://mano.is.tue.mpg.de/ and place under `data/mano/`.

### 5. Off-the-shelf checkpoints (~16 GB)

```bash
bash scripts/download_checkpoints.sh checkpoints
```

The script caches WiLoR via `wilor_mini` and prints upstream URLs for HaMeR,
HandOccNet, and Mesh Graphormer. Follow each upstream README for the
final download step (Drive links, `download_demo.sh`, etc.).

## Reproducing each result

Every row in the headline table maps to one or two `Makefile` targets.
`make help` lists every target.

| Result                                          | Targets                                           | Wall-clock on A100 |
|-------------------------------------------------|---------------------------------------------------|--------------------|
| Off-shelf benchmark                             | `eval-{wilor,hamer,handoccnet,mgfm}-{pov,aria}`    | ~30 min total      |
| Per-sequence Aria breakdown                     | `eval-per-seq`                                     | ~10 min            |
| Per-finger PA-MPJPE                             | `eval-per-finger`                                  | ~5 min             |
| Exp A (POV-only FT)                             | `ft-wilor-pov` then `eval-wilor-{pov,aria}`        | ~3 h               |
| **Exp B (mixed FT, headline)**                  | `ft-wilor-mixed` then `eval-wilor-{pov,aria}`      | ~3 h               |
| Exp B+ (anchored regularizer)                   | `ft-wilor-anchored`                                | ~3 h               |
| Exp E/F (HandOccNet FT, expected to regress)    | `ft-honet`                                         | ~2 h               |
| Inference-time ensemble                         | `eval-ensemble`                                    | ~20 min            |
| **Exp I (multi-teacher distillation)**          | `precompute-teacher` then `ft-wilor-distill`       | ~6 h               |

Smoke test (should print PA-MPJPE around 11 mm):

```bash
make eval-wilor-pov DATA=data CKPT=checkpoints RESULTS=results
```

## Path overrides

Defaults assume `/workspace/...` layout from the original training
environment. Two ways to redirect:

```bash
# Option 1: Makefile arguments
make eval-wilor-pov DATA=/mnt/datasets CKPT=/mnt/ckpt RESULTS=/mnt/out

# Option 2: environment variables (read by every src/ script)
export AIM2_ARIA_VAL_DIR=/path/to/aria_val/data
export AIM2_ARIA_DIR=/path/to/aria_val/data
export AIM2_POV_DIR=/path/to/pov_surgery/data
export AIM2_SIDECAR_DIR=/path/to/phase0_sidecars
export AIM2_CACHE_DIR=/path/to/cache
```

Python scripts also accept `--data`, `--ckpt`, `--out` flags directly.

## Reproducibility notes

* **Custom fine-tuning checkpoints are not redistributed.** Re-run training
  with the `make ft-*` targets. Roughly 30 GPU-hours of A100 time covers
  the full ablation matrix (Exp A, B, B+, C, D, E, F, I).
* **POV-Surgery** is publicly released by Wang et al. (2023) at
  https://batfacewayne.github.io/POV_Surgery_io/.
* **Aria-Surgical-Hand-Pose** is privately collected at Beth Israel
  Hospital. Access by request from the report authors.

## Citing this work

Cite the final report. The report's bibliography lists every model and
dataset used here (WiLoR, HaMeR, HandOccNet, Mesh Graphormer, POV-Surgery,
MANO, Aria).

## License

Code in `src/`, `scripts/`, and the `Makefile` is released under the MIT
license (see `LICENSE`). Forked upstream model repositories under
`external/` retain their own licenses.
