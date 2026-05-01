# Setup

End-to-end setup from a fresh machine: Python dependencies, datasets, model
checkpoints, and the seven external repos that house the off-shelf models.

> **System notes.** GPU strongly recommended. POV-Surgery requires ~17 GB of
> disk; the Aria val data is ~3 GB; off-shelf checkpoints add another ~16 GB;
> full custom fine-tuning checkpoints are about 115 GB. The reference training
> environment is a single A100; full ablation matrix takes ~30 GPU-hours.

## 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Tested with Python 3.10, CUDA 12.1, PyTorch 2.3.

## 2. External model repositories

The four benchmarked models live in their own upstream repositories. We pin
exact commit SHAs (recorded in `external/README.md`) and apply a single
50-line patch to HaMeR. Follow the commands in `external/README.md` once to
clone all seven repos and apply the patch.

## 3. Datasets

### POV-Surgery (public; ~17 GB)

POV-Surgery is publicly distributed via the project page:
https://batfacewayne.github.io/POV_Surgery_io/

`scripts/download_pov.sh` documents the expected on-disk layout. After
following the project page's instructions, the dataset should land under
`data/pov_surgery/` with the expected subfolders (`annotation/`,
`handoccnet_train/`, `tool_mesh/`, `mask/`, `mask_blender/`, `color/`, plus
the `POV_Surgery_info.csv`).

### Aria-Surgical-Hand-Pose (private; ~3 GB)

This dataset was collected at Beth Israel Hospital by the authors and is
**not publicly distributed**. Access is granted on request through a private
HuggingFace folder.

The dataset has two parallel ground-truth signals per frame:

* **Aria-MPS native**: 21 OP-order joints from Aria's Machine Perception
  Services multi-camera reconstruction. Treated as gold-standard real-OR
  ground truth. Held out from all training.
* **HaMeR-SAM (HSAM) pseudo**: 21 MANO joints + 778 mesh vertices, generated
  by running HaMeR with SAM-derived masks and then manually checked and
  filtered frame-by-frame. Used as a fine-tuning target.

A "v2-clean" filter on the raw 3,259-frame collection drops 519 reject/skip
keys to give the canonical n=2,333 evaluation set used throughout the paper.

If you have access, place the dataset under `data/aria_val/` matching the
layout returned by the HuggingFace download.

### MANO model files (~258 MB)

Download `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` from the MANO project page
(https://mano.is.tue.mpg.de/, registration required) and place them under
`data/mano/`.

## 4. Model checkpoints

### Off-the-shelf (~16 GB total)

```bash
bash scripts/download_checkpoints.sh checkpoints/
```

This caches WiLoR via `wilor_mini` and prints upstream URLs for HaMeR,
HandOccNet, and Mesh Graphormer. Follow each upstream README for the final
download step (Google Drive links / `download_demo.sh`).

### Fine-tuned (custom; ~115 GB if you keep them all)

Custom fine-tuning checkpoints are **not redistributed**. Reproduce them with
the `make ft-*` targets in the `Makefile`:

```bash
make ft-wilor-pov          # Exp A
make ft-wilor-mixed        # Exp B (headline)
make ft-wilor-anchored     # Exp B+
make ft-wilor-distill      # Exp I
make ft-honet              # Exp E/F (HandOccNet FT, expected to regress)
```

Each run writes its checkpoint into `checkpoints/<run_name>/`.

## 5. Smoke test

```bash
make eval-wilor-pov DATA=data CKPT=checkpoints RESULTS=results
```

Should print PA-MPJPE around **11 mm** on POV-Surgery (off-shelf WiLoR). If
you see that number, the pipeline is wired correctly.

## 6. Path overrides

Every `Makefile` target reads three environment variables / make arguments:

| Variable | What it controls | Default |
|----------|------------------|---------|
| `DATA`     | dataset root      | `data`         |
| `CKPT`     | checkpoint root   | `checkpoints`  |
| `RESULTS`  | output JSONs root | `results`      |

```bash
make eval-wilor-pov DATA=/mnt/datasets CKPT=/mnt/ckpt RESULTS=/mnt/out
```

If you bypass the Makefile and run `python -m src.eval.eval_wilor_runner`
directly, every script accepts the same paths via `--data`, `--ckpt`, `--out`
(and similar) flags. Hardcoded `/workspace/...` paths visible in some
`argparse` defaults are the original training-environment defaults; they are
overridable via the flags above.
