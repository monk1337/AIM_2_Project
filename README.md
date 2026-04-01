# Gloved Hands, Blind Models: Bridging the Domain Gap in Surgical Hand Pose Estimation

**BMI 702 / AIM II: Artificial Intelligence in Medicine II**
Harvard Department of Biomedical Informatics, Spring 2026

**Team:** Ankit Pal, Yuchang Su

**Instructor:** Marinka Zitnik, PhD

## Weekly Progress Log

### Week 3 (due Feb 11)
- Defined project scope: exploring AI applications in medical robotics
- Surveyed VLA model landscape and surgical robotics simulators
- Identified VITRA as the target VLA framework with MANO-based action space

### Week 4 (due Feb 18)
- Began studying VLA models for surgical adaptation
- Reviewed VITRA paper (arxiv:2505.10251) for hand pose action space design
- Started exploring POV-Surgery dataset for MANO ground truth
- Submitted Project Proposal (Feb 20)

### Week 5 (due Feb 25)
- Built initial WiLoR evaluation prototype on POV-Surgery
- Implemented MANO forward kinematics and camera coordinate transforms
- Discovered WiLoR's YOLO detector achieves 0% detection on synthetic surgical hands
- Evaluated VLMs (Gemini, Claude, GPT) on surgical frame understanding for annotation
- Developed full evaluation pipeline with 5 metrics (MPJPE, PA-MPJPE, PVE, PA-PVE, P2D)
- Added per-finger analysis and visualization scripts
- **Key pivot:** Shifted focus from VLA adaptation to systematic hand pose evaluation

### Week 6 (due Mar 4)
- Obtained access to EgoSurgery-Phase real OR dataset
- Set up data infrastructure for the large-scale dataset
- Added HaMeR evaluation pipeline with weak-perspective camera conversion
- Resolved coordinate space alignment challenges (HaMeR MANO-local vs. OpenCV camera frame)
- Found both models converge to approximately 11mm PA-MPJPE, confirming domain gap ceiling

### Week 7 (due Mar 11)
- Extended evaluation to 141 real OR frames from EgoSurgery-Phase
- Built stratified sampling pipeline (equal phase allocation, video diversity, quality filtering)
- Implemented side-by-side WiLoR vs. HaMeR overlay visualizations
- Discovered asymmetric domain gap: 81.6% YOLO detection on real data vs. 0% on synthetic
- Built data processing pipeline (frame filtering, hand pose extraction, episode building)

### Spring Break (Mar 18, no class)

### Week 8 (due Mar 25)
- Scaled WiLoR and HaMeR evaluation to full POV-Surgery test set (26,418 frames, 17 sequences)
- Cleaned up pipeline code and documentation
- Prepared midterm presentation and report
- Generated EgoSurg-Bench data collection plan for future dataset recording

### Midterm (Mar 23 presentation, Mar 27 submission)
- Presented at midterm session (Mar 23, TMEC 227)
- Submitted midterm report: "Gloved Hands, Blind Models"
- Documented complete Phase 1 (off-the-shelf baselines) and Phase 2 (real data evaluation) results

### Week 9 (due Apr 1) -- Current
- Begin Phase 3: domain adaptation via fine-tuning WiLoR on POV-Surgery

### Upcoming
- **Weeks 10-11 (Apr 8 - Apr 15):** Fine-tuning experiments with data efficiency curve and frozen vs. unfrozen backbone
- **Weeks 12-13 (Apr 22 - Apr 29):** Error taxonomy, cross-domain transfer evaluation, final analysis
- **Apr 29:** Final Presentations (Countway 403)
- **May 1:** Project Final Submission

## Project Overview

Vision-language-action (VLA) models for surgical robotics need reliable hand pose estimation to define their action space. Modern hand pose estimators use the MANO parametric hand model (61 parameters per hand) to predict 3D hand meshes from single RGB images. However, their performance on surgical data is unknown.

When we attempted to build a surgical VLA pipeline using MANO-based hand parameters as the 102-dimensional bimanual action space, we discovered a fundamental upstream bottleneck: state-of-the-art hand pose estimators fail substantially on egocentric surgical imagery. This finding redirected our project from VLA adaptation toward a systematic evaluation of the domain gap in surgical hand pose estimation.

We present a benchmark evaluation of two leading MANO-based hand pose estimators (WiLoR and HaMeR) on surgical data spanning synthetic and real operating room footage, quantifying where and why they fail, and proposing a domain adaptation strategy to close the gap.

## Motivation

Surgical settings introduce domain challenges absent from standard hand pose training data:

| Challenge | Everyday Data | Surgical Data |
|-----------|--------------|---------------|
| Skin visibility | Bare hands, skin texture | Latex/nitrile gloves, uniform color |
| Hand-object | Cups, phones | Scalpels, forceps, needle drivers |
| Viewpoint | Third-person or mild ego | Extreme egocentric, self-occlusion |
| Lighting | Controlled indoor | Variable OR lights, instrument reflections |
| Training data | Millions of annotations | Near zero annotated surgical images |

## Research Questions

1. **How bad are current methods?** Quantify off-the-shelf performance on surgical data with ground truth.
2. **Why do they fail?** Decompose errors into position vs. shape, per-finger, and temporal components.
3. **Can domain adaptation fix it?** Fine-tune on synthetic surgical data and measure improvement.

## Models Evaluated

| Model | Architecture | Key Details |
|-------|-------------|-------------|
| **WiLoR** (Potamias et al., CVPR 2025) | YOLO detector + ViT backbone + multi-scale refinement | Two-stage pipeline, includes built-in hand detector, 5.5mm PA-MPJPE on FreiHAND |
| **HaMeR** (Pavlakos et al., CVPR 2024) | ViT-H (1280-dim, 32 layers) + transformer decoder | Single-stage, no detector, 2.7GB checkpoint, 6.0mm PA-MPJPE on FreiHAND |

## Datasets

| Dataset | Domain | Frames | Ground Truth | Evaluation |
|---------|--------|--------|-------------|------------|
| **POV-Surgery** (Wang et al., 2023) | Synthetic egocentric surgery | 26,418 (test set, 17 sequences) | Per-frame MANO parameters | Quantitative |
| **EgoSurgery-Phase** (Fujii et al., 2024) | Real egocentric OR video | 141 (stratified sample) | Hand bounding boxes (COCO format) | Qualitative |

## Evaluation Protocol

We evaluate in **crop-regress mode**: for each frame, we derive a padded bounding box (1.5x scale) from ground-truth annotations and feed the cropped hand region directly to the estimator. This isolates regression quality from detection quality.

**Metrics computed:**
- MPJPE: Mean per-joint position error (root-relative), in mm
- PA-MPJPE: Procrustes-aligned MPJPE (isolates shape error), in mm
- PVE: Per-vertex error on 778-vertex MANO mesh, in mm
- PA-PVE: Procrustes-aligned PVE, in mm
- P2D: 2D reprojection error in full image coordinates, in px

## Key Results

### POV-Surgery Quantitative Results (26,418 frames, 17 sequences)

| Method | MPJPE (mm) | PA-MPJPE (mm) | PVE (mm) | PA-PVE (mm) | P2D (px) |
|--------|-----------|---------------|---------|-------------|---------|
| WiLoR (off-the-shelf) | 50.36 | **10.69** | **47.92** | **10.01** | **26.00** |
| HaMeR (off-the-shelf) | 54.31 | 11.17 | 51.68 | 10.48 | 30.58 |

### Detection on Surgical Data

| Setting | WiLoR YOLO Detection Rate |
|---------|--------------------------|
| POV-Surgery (synthetic) | **0%** (0/1,571 frames) |
| EgoSurgery-Phase (real OR) | **81.6%** (115/141 frames) |

### Key Findings

1. **Domain gap ceiling, not model capacity.** Both models converge to approximately 11mm PA-MPJPE despite very different architectures (4.3% difference). This is roughly 1.8x worse than standard benchmarks (approximately 6mm), confirming a domain shift problem.

2. **Position dominates shape error.** Approximately 79% of total MPJPE comes from global translation errors. The models understand hand structure (only approximately 21% is shape/articulation error) but misplace the hand in 3D space.

3. **Asymmetric domain gap.** WiLoR's YOLO detector achieves 0% detection on synthetic data but 81.6% on real OR data. The synthetic-to-real gap affects detection and regression differently. For deployment on real surgical video, the detector may work off-the-shelf, but pose regression still requires domain adaptation.

4. **Per-finger error pattern.** Index and middle fingers show the highest error (approximately 60-65mm MPJPE) for both models, consistent with them being the most articulated fingers during surgical tool manipulation and most occluded by instruments.

5. **Correlated temporal errors.** Both models exhibit nearly identical temporal error patterns, with error peaks at the same challenging frames. This further supports domain-driven rather than model-specific error.

## Proposed Next Steps: Domain Adaptation via Fine-Tuning

Our baseline evaluation establishes that the approximately 11mm PA-MPJPE ceiling is a domain gap problem. The proposed method is to close this gap through supervised fine-tuning on synthetic surgical data.

**Strategy:** Fine-tune WiLoR on the full POV-Surgery training set (approximately 70K frames with MANO ground truth).

**Planned experiments:**
1. Data efficiency curve: Fine-tune on 10%, 25%, 50%, and 100% of training data
2. Frozen vs. unfrozen backbone: Compare ViT backbone frozen (head-only) vs. full fine-tuning
3. Before/after evaluation: Same metrics, same frames, direct comparison
4. Cross-domain transfer: Evaluate fine-tuned model on real EgoSurgery-Phase data

**Target:** Reduce PA-MPJPE from approximately 11mm toward approximately 4mm (matching published fine-tuned results).

## Repository Structure

```
AIM_2_Project/
|
|-- eval/                           # Core evaluation pipeline
|   |-- run_eval_wilor.py           # WiLoR evaluation on POV-Surgery (detect + crop modes)
|   |-- run_eval_hamer.py           # HaMeR evaluation on POV-Surgery (crop mode)
|   |-- run_eval_handoccnet.py      # HandOCCNet evaluation on POV-Surgery
|   |-- run_eval_egosurgery.py      # Qualitative eval on real EgoSurgery-Phase data
|   |-- visualize_results.py        # Generate publication figures from results JSON
|   |-- visualize_predictions.py    # GT vs predicted skeleton overlay visualizations
|   |-- requirements.txt            # Python dependencies
|   +-- README.md                   # Evaluation pipeline documentation
|
|-- scripts/                        # Data processing and VLA pipeline
|   |-- step1_filter_frames.py      # Filter EgoSurgery frames by phase and hand visibility
|   |-- step1b_sample_windows.py    # Find consecutive temporal windows for episodes
|   |-- step2_extract_hand_pose.py  # Extract MANO params via WiLoR on EgoSurgery
|   |-- step2b_extract_window_poses.py  # Batch pose extraction for temporal windows
|   |-- step3_build_episodes.py     # Build VITRA-compatible .npy episodes
|   |-- eval_wilor_pov_surgery.py   # Early WiLoR evaluation prototype
|   +-- generate_data_collection_doc.py  # EgoSurg-Bench data collection plan generator
|
|-- hamer/                          # HaMeR-specific evaluation and visualization
|   |-- evaluate.py                 # HaMeR evaluation script
|   |-- evaluate_hamer.py           # Extended HaMeR evaluation
|   |-- visualize_hamer.py          # HaMeR prediction visualization
|   |-- visualize_crop.py           # Cropped overlay visualization
|   +-- setup_env.sh                # Environment setup for HaMeR
|
+-- README.md                       # This file
```

## Setup and Reproduction

### Prerequisites

- Python 3.12+
- PyTorch 2.0+
- MANO body model files (download from [MANO website](https://mano.is.tue.mpg.de/))

### Data Setup

**POV-Surgery:** Download from the [POV-Surgery repository](https://github.com/Linguanbiao/POV-Surgery). Place under `pov_surgery_data/`.

**EgoSurgery-Phase:** Request access from the dataset authors (Fujii et al., MICCAI 2024). Place under `full_data/`.

### Running Evaluations

```bash
# Activate environment
source .venv/bin/activate
cd eval/

# WiLoR evaluation (crop-regress mode, full test set)
python run_eval_wilor.py --mode crop --split full --data-dir ../pov_surgery_data

# HaMeR evaluation (crop-regress mode, full test set)
python run_eval_hamer.py --mode crop --split full --data-dir ../pov_surgery_data

# EgoSurgery qualitative evaluation (both models, 141 frames)
python run_eval_egosurgery.py --mode crop --model both --data-dir ../full_data

# Generate figures
python visualize_results.py --results results/full/wilor_crop_results.json
python visualize_results.py --results results/full/hamer_crop_results.json --output-dir results/figures_hamer/

# Quick test (10 frames)
python run_eval_wilor.py --mode crop --max-frames 10 --data-dir ../pov_surgery_data
```

## Challenges and Mitigations

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| Hand detection failure | WiLoR YOLO achieves 0% on synthetic surgical hands | Evaluate in crop-regress mode using GT bounding boxes |
| Platform compatibility | HaMeR uses float64 ops unsupported by Apple MPS | Force CPU execution for HaMeR inference |
| Coordinate space alignment | HaMeR outputs in MANO-local coordinates, GT is in OpenCV camera frame | Implemented full weak-perspective to full-image conversion |
| Project pivot | Original VLA proposal required working hand tracking | Refocused on measurable hand pose evaluation with clear baselines |

## Member Contributions

| Member | Contributions |
|--------|-------------|
| Ankit Pal | WiLoR evaluation pipeline, EgoSurgery data processing and sampling, background research, experimental design, report writing |
| Yuchang Su | HaMeR evaluation pipeline, POV data processing, literature survey, visualization scripts, result analysis, report writing |

## References

- Potamias et al. "WiLoR: End-to-end 3D hand localization and reconstruction in-the-wild." CVPR, 2025.
- Pavlakos et al. "Reconstructing hands in 3D with transformers." CVPR, 2024.
- Wang et al. "POV-Surgery: A dataset for egocentric hand and tool pose estimation during surgical activities." ICCV Workshop, 2023.
- Fujii et al. "EgoSurgery-Phase: A dataset of surgical phase recognition from egocentric open surgery videos." MICCAI, 2024.
- Romero et al. "Embodied hands: Modeling and capturing hands and bodies together." SIGGRAPH Asia, 2017.
- Li et al. "VITRA: Scalable vision-language-action model pretraining for robotic manipulation." arXiv:2510.21571, 2025.
