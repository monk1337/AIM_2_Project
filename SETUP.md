# Setup

End-to-end setup from a fresh machine: dependencies, datasets, model
checkpoints, and the seven external repos that house the off-shelf models.

> Note: GPU strongly recommended. POV-Surgery + Aria-val together require
> roughly 25 GB of disk; off-shelf checkpoints add another 16 GB; full custom
> fine-tuning checkpoints are about 115 GB.

## 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. External model repositories

The four benchmarked models live in their own upstream repos. We pin exact
commits and apply a single small patch to HaMeR. Run the script in
`external/README.md` once to clone everything and apply the patch.

## 3. Datasets

### POV-Surgery (synthetic egocentric surgery; 17 GB)

Public on HuggingFace.

```bash
bash scripts/download_pov.sh data/pov_surgery
```

### Aria-Surgical-Hand-Pose (real-OR validation; 3.3 GB + 9 MB sidecars)

```bash
bash scripts/download_aria.sh data/aria_val
```

The script also pulls the lab-canonical filter sidecars
(`reject_keys_all_20260419.json`, `mps_v2_val_20260419.json`) from
`aaditya/phase0-artifacts`. These are required for the v2-clean filter that
reduces the raw 3,259 frames to the canonical 2,333.

### MANO model files (258 MB)

Download `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` from the MANO project page
(registration required) and place them in `data/mano/`.

## 4. Model checkpoints

### Off-the-shelf (16 GB total)

```bash
bash scripts/download_checkpoints.sh checkpoints/
```

This caches WiLoR via `wilor_mini` and prints upstream URLs for HaMeR,
HandOccNet, and Mesh Graphormer. Follow each upstream's README for the final
download step (Drive links / `download_demo.sh`).

### Fine-tuned (custom)

Custom fine-tuning checkpoints are not redistributed. Reproduce them with the
`make ft-*` targets in the `Makefile`. Roughly 30 hours of A100 time for the
full ablation matrix.

## 5. Quick smoke test

```bash
make eval-wilor-pov-quick   # runs WiLoR off-shelf on 100 POV frames
```

If this prints a PA-MPJPE around 11 mm, the pipeline is wired correctly.
