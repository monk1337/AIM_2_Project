#!/usr/bin/env python3
"""Evaluate WiLoR and HaMeR on EgoSurgery-Phase (real egocentric OR video).

EgoSurgery has hand bounding box annotations but NO MANO ground truth,
so this is a qualitative evaluation: detection rates, visual overlays,
and failure categorization.

Modes:
  --mode detect:  Use WiLoR's YOLO detector (tests detection on real OR)
  --mode crop:    Use GT hand bboxes from COCO annotations (tests regression only)

Usage:
    python run_eval_egosurgery.py --mode detect --data-dir ../full_data --n-frames 200
    python run_eval_egosurgery.py --mode crop --data-dir ../full_data --n-frames 200
    python run_eval_egosurgery.py --mode crop --data-dir ../full_data --n-frames 200 --model hamer
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Hand skeleton connections (OpenPose order, 21 joints)
SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
]


def load_coco_annotations(bbox_dir):
    """Load COCO-format hand bounding box annotations for all videos.

    Returns dict: {frame_filename: [{'bbox': [x,y,w,h], 'category_id': int, 'category_name': str}, ...]}
    """
    categories = {
        1: "own_left", 2: "own_right",
        3: "other_left", 4: "other_right",
    }

    all_annotations = {}
    hands_dir = bbox_dir / "by_video" / "hands"
    if not hands_dir.exists():
        print(f"ERROR: No hand annotations at {hands_dir}")
        return {}

    for vid_dir in sorted(hands_dir.iterdir()):
        if not vid_dir.is_dir() or vid_dir.name.startswith("."):
            continue
        ann_path = vid_dir / "annotations.json"
        if not ann_path.exists():
            continue

        with open(ann_path) as f:
            data = json.load(f)

        # Build image_id -> filename mapping
        id_to_fname = {img["id"]: img["file_name"] for img in data["images"]}

        # Update category names from this file if available
        if "categories" in data:
            for cat in data["categories"]:
                categories[cat["id"]] = cat["name"].lower().replace(" ", "_")

        for ann in data["annotations"]:
            fname = id_to_fname.get(ann["image_id"])
            if fname is None:
                continue
            if fname not in all_annotations:
                all_annotations[fname] = []
            all_annotations[fname].append({
                "bbox": ann["bbox"],  # [x, y, w, h] COCO format
                "category_id": ann["category_id"],
                "category_name": categories.get(ann["category_id"], "unknown"),
            })

    return all_annotations


def sample_frames(windows_path, n_frames, annotations, min_bbox_area=5000, seed=42):
    """Sample diverse frames from bimanual windows, stratified by phase.

    Strategy:
    - Equal phase representation (not proportional, incision is rare but interesting)
    - 1 frame per window to avoid temporal redundancy
    - Diverse videos within each phase
    - Filter out frames with tiny bboxes (area < min_bbox_area)
    """
    with open(windows_path) as f:
        data = json.load(f)
    windows = data["windows"]
    rng = random.Random(seed)

    # For each window, pick one representative frame (middle of window)
    # and attach metadata
    candidates_by_phase = defaultdict(list)
    for w in windows:
        fids = w["frame_ids"]
        # Pick middle frame to get representative pose
        mid_fid = fids[len(fids) // 2]
        # Check bbox quality
        fname = f"{mid_fid}.jpg"
        anns = annotations.get(fname, [])
        own_anns = [a for a in anns if a["category_id"] in (1, 2)]
        if len(own_anns) < 2:
            # Not bimanual in annotations, try another frame
            for alt_fid in fids:
                alt_fname = f"{alt_fid}.jpg"
                alt_anns = annotations.get(alt_fname, [])
                alt_own = [a for a in alt_anns if a["category_id"] in (1, 2)]
                if len(alt_own) >= 2:
                    mid_fid = alt_fid
                    own_anns = alt_own
                    break
            else:
                continue  # No bimanual frame in this window

        # Filter by minimum bbox area
        areas = [a["bbox"][2] * a["bbox"][3] for a in own_anns]
        if any(a < min_bbox_area for a in areas):
            continue

        candidates_by_phase[w["phase"]].append({
            "frame_id": mid_fid,
            "video_id": w["video_id"],
            "phase": w["phase"],
            "window_id": w["window_id"],
            "n_own_hands": len(own_anns),
            "has_other_hands": any(a["category_id"] in (3, 4) for a in anns),
        })

    # Equal allocation per phase
    phases = sorted(candidates_by_phase.keys())
    n_per_phase = n_frames // len(phases)

    sampled = []
    for phase in phases:
        candidates = candidates_by_phase[phase]

        if len(candidates) <= n_per_phase:
            sampled.extend(candidates)
            continue

        # Maximize video diversity: round-robin across videos
        by_video = defaultdict(list)
        for c in candidates:
            by_video[c["video_id"]].append(c)

        phase_sampled = []
        video_ids = sorted(by_video.keys())
        rng.shuffle(video_ids)

        # Round-robin until we have enough
        idx = 0
        while len(phase_sampled) < n_per_phase:
            vid = video_ids[idx % len(video_ids)]
            if by_video[vid]:
                frame = rng.choice(by_video[vid])
                by_video[vid].remove(frame)
                phase_sampled.append(frame)
            else:
                video_ids.remove(vid)
                if not video_ids:
                    break
            idx += 1

        sampled.extend(phase_sampled)

    # Fill remaining slots from any phase
    remaining_budget = n_frames - len(sampled)
    if remaining_budget > 0:
        all_remaining = []
        sampled_ids = {s["frame_id"] for s in sampled}
        for phase in phases:
            for c in candidates_by_phase[phase]:
                if c["frame_id"] not in sampled_ids:
                    all_remaining.append(c)
        rng.shuffle(all_remaining)
        sampled.extend(all_remaining[:remaining_budget])

    return sampled


def bbox_xywh_to_xyxy(bbox):
    """Convert COCO [x,y,w,h] to [x1,y1,x2,y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def get_own_hand_bboxes(annotations, frame_fname):
    """Get own-hand bounding boxes for a frame. Returns list of (bbox_xyxy, is_right)."""
    anns = annotations.get(frame_fname, [])
    hands = []
    for ann in anns:
        cat = ann["category_name"]
        if "own" not in cat:
            continue
        bbox_xyxy = bbox_xywh_to_xyxy(ann["bbox"])
        is_right = "right" in cat
        hands.append((bbox_xyxy, is_right))
    return hands


def pad_bbox(bbox_xyxy, img_h, img_w, pad_factor=1.5):
    """Pad a bounding box by pad_factor, clamp to image bounds."""
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    new_w, new_h = w * pad_factor, h * pad_factor
    nx1 = max(0, cx - new_w / 2)
    ny1 = max(0, cy - new_h / 2)
    nx2 = min(img_w, cx + new_w / 2)
    ny2 = min(img_h, cy + new_h / 2)
    return [nx1, ny1, nx2, ny2]


def draw_skeleton_2d(img, keypoints_2d, color=(0, 255, 0), thickness=2, radius=3):
    """Draw 21-joint hand skeleton on image."""
    for j1, j2 in SKELETON:
        pt1 = tuple(int(v) for v in keypoints_2d[j1])
        pt2 = tuple(int(v) for v in keypoints_2d[j2])
        cv2.line(img, pt1, pt2, color, thickness)
    for j in range(21):
        pt = tuple(int(v) for v in keypoints_2d[j])
        cv2.circle(img, pt, radius, color, -1)
    return img


def draw_bbox(img, bbox_xyxy, color=(255, 255, 0), thickness=2, label=None):
    """Draw bounding box on image."""
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    return img


# ── WiLoR ──────────────────────────────────────────────────────────────

def load_wilor(device, dtype):
    """Load WiLoR pipeline."""
    _original_load = torch.load
    def _patched_load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _original_load(*a, **kw)
    torch.load = _patched_load

    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    torch.load = _original_load
    return pipe


def run_wilor_detect(pipe, image):
    """Run WiLoR with its own YOLO detector. Returns list of detections."""
    outputs = pipe.predict(image)
    results = []
    for out in outputs:
        kp2d = out["wilor_preds"].get("pred_keypoints_2d")
        if kp2d is not None:
            results.append({
                "keypoints_2d": kp2d[0].copy(),
                "is_right": out["is_right"] == 1.0,
                "confidence": 1.0,
            })
    return results


def run_wilor_crop(pipe, image, bboxes_xyxy, is_rights):
    """Run WiLoR with provided bounding boxes. Returns list of detections."""
    bboxes = np.array(bboxes_xyxy, dtype=np.float32)
    is_rights_float = [1.0 if r else 0.0 for r in is_rights]
    outputs = pipe.predict_with_bboxes(image, bboxes, is_rights_float, rescale_factor=2.5)
    results = []
    for out in outputs:
        kp2d = out["wilor_preds"].get("pred_keypoints_2d")
        if kp2d is not None:
            results.append({
                "keypoints_2d": kp2d[0].copy(),
                "is_right": out["is_right"] == 1.0,
                "confidence": 1.0,
            })
    return results


# ── HaMeR ──────────────────────────────────────────────────────────────

def load_hamer_model(device):
    """Load HaMeR model."""
    project_root = Path(__file__).resolve().parent.parent
    import hamer.configs
    hamer.configs.CACHE_DIR_HAMER = str(project_root / "_DATA")
    from hamer.models import load_hamer

    hamer_ckpt = project_root / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
    model, model_cfg = load_hamer(str(hamer_ckpt))
    model.to(device).eval()
    return model, model_cfg


def run_hamer_crop(model, model_cfg, image_bgr, bboxes_xyxy, is_rights, device):
    """Run HaMeR with provided bounding boxes. Returns list of detections with 2D keypoints."""
    from hamer.datasets.vitdet_dataset import ViTDetDataset

    results = []
    for bbox_xyxy, is_right in zip(bboxes_xyxy, is_rights):
        boxes = np.array([[bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]]])
        right = np.array([1.0 if is_right else 0.0])

        dataset = ViTDetDataset(model_cfg, image_bgr, boxes, right, rescale_factor=2.5)
        if len(dataset) == 0:
            continue

        batch = dataset[0]
        batch_t = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_t[k] = v.unsqueeze(0).to(device)
            elif isinstance(v, np.ndarray):
                batch_t[k] = torch.from_numpy(v).unsqueeze(0).to(device)
            else:
                batch_t[k] = v

        with torch.no_grad():
            out = model(batch_t)

        # Convert weak-perspective camera to full-image 2D coordinates
        # Using HaMeR's cam_crop_to_full formula (renderer.py):
        #   bs = box_size * pred_cam[0]
        #   tz = 2 * scaled_fl / bs
        #   tx = (2*(cx - w/2) / bs) + pred_cam[1]
        #   ty = (2*(cy - h/2) / bs) + pred_cam[2]
        pred_cam = out["pred_cam"][0].cpu().numpy()
        pred_joints = out["pred_keypoints_3d"][0].cpu().numpy()  # (21, 3) in MANO local space
        box_center = batch["box_center"].copy()
        img_w, img_h = batch["img_size"].copy()
        box_size = float(batch["box_size"])

        scaled_fl = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(img_w, img_h)

        # Use box_size (not MODEL.IMAGE_SIZE) as the denominator, matches cam_crop_to_full
        bs = box_size * pred_cam[0] + 1e-9
        cam_t_full = np.array([
            (2 * (box_center[0] - img_w / 2) / bs) + pred_cam[1],
            (2 * (box_center[1] - img_h / 2) / bs) + pred_cam[2],
            2 * scaled_fl / bs,
        ])

        joints_cam = pred_joints + cam_t_full[None, :]
        pred_2d = np.zeros((21, 2))
        pred_2d[:, 0] = scaled_fl * joints_cam[:, 0] / joints_cam[:, 2] + img_w / 2
        pred_2d[:, 1] = scaled_fl * joints_cam[:, 1] / joints_cam[:, 2] + img_h / 2

        results.append({
            "keypoints_2d": pred_2d,
            "is_right": is_right,
            "confidence": 1.0,
        })

    return results


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate hand pose on EgoSurgery-Phase")
    parser.add_argument("--mode", choices=["detect", "crop"], default="crop",
                        help="detect: WiLoR YOLO detector; crop: GT bboxes (default: crop)")
    parser.add_argument("--model", choices=["wilor", "hamer", "both"], default="both",
                        help="Which model to evaluate (default: both)")
    parser.add_argument("--n-frames", type=int, default=200,
                        help="Number of frames to sample (default: 200)")
    parser.add_argument("--data-dir", type=str, default="../full_data",
                        help="Path to EgoSurgery full_data/ directory")
    parser.add_argument("--windows-path", type=str, default=None,
                        help="Path to windows.json (default: ../pipeline_output/step1b_windows/windows.json)")
    parser.add_argument("--output-dir", type=str, default="results/egosurgery",
                        help="Output directory (default: results/egosurgery)")
    parser.add_argument("--device", type=str, default=None, help="Device (mps/cuda/cpu)")
    parser.add_argument("--save-overlays", action="store_true", default=True,
                        help="Save per-frame overlay images")
    parser.add_argument("--no-overlays", action="store_true",
                        help="Skip saving overlay images")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    if not args.no_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    if args.windows_path:
        windows_path = Path(args.windows_path).resolve()
    else:
        windows_path = Path(__file__).resolve().parent.parent / "pipeline_output" / "step1b_windows" / "windows.json"

    images_dir = data_dir / "images"
    bbox_dir = data_dir / "annotations" / "bbox"

    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)

    # Device setup
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = torch.float32

    # ── Load annotations ──────────────────────────────────────────────
    print("Loading COCO bbox annotations...")
    annotations = load_coco_annotations(bbox_dir)
    print(f"  Loaded annotations for {len(annotations)} frames")

    # ── Sample frames ─────────────────────────────────────────────────
    print(f"Sampling {args.n_frames} frames from bimanual windows...")
    sampled = sample_frames(windows_path, args.n_frames, annotations)
    print(f"  Sampled {len(sampled)} frames across "
          f"{len(set(f['video_id'] for f in sampled))} videos, "
          f"{len(set(f['phase'] for f in sampled))} phases")

    # Phase/video distribution
    phase_counts = defaultdict(int)
    video_counts = defaultdict(int)
    for f in sampled:
        phase_counts[f["phase"]] += 1
        video_counts[f["video_id"]] += 1
    print(f"  Phases: {dict(phase_counts)}")

    # ── Load models ───────────────────────────────────────────────────
    models_to_run = []

    if args.model in ("wilor", "both"):
        print("Loading WiLoR...")
        wilor_pipe = load_wilor(device, dtype)
        models_to_run.append("wilor")
        print("  WiLoR loaded.")

    if args.model in ("hamer", "both"):
        if args.mode == "detect":
            print("WARNING: HaMeR has no built-in detector. Skipping HaMeR in detect mode.")
        else:
            print("Loading HaMeR...")
            # HaMeR must run on CPU (float64 issue with MPS)
            hamer_device = torch.device("cpu")
            hamer_model, hamer_cfg = load_hamer_model(hamer_device)
            models_to_run.append("hamer")
            print("  HaMeR loaded (CPU).")

    # ── Evaluation loop ───────────────────────────────────────────────
    all_results = []
    t_start = time.time()

    for i, frame_info in enumerate(sampled):
        fid = frame_info["frame_id"]
        vid = frame_info["video_id"]
        phase = frame_info["phase"]
        img_path = images_dir / vid / f"{fid}.jpg"

        if not img_path.exists():
            all_results.append({
                "frame_id": fid, "video_id": vid, "phase": phase,
                "error": "image_not_found",
            })
            continue

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(sampled)}]  {fps:.1f} frames/s")

        image_rgb = np.array(Image.open(img_path).convert("RGB"))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        img_h, img_w = image_rgb.shape[:2]

        # Get GT hand bboxes
        frame_fname = f"{fid}.jpg"
        gt_hands = get_own_hand_bboxes(annotations, frame_fname)

        frame_result = {
            "frame_id": fid,
            "video_id": vid,
            "phase": phase,
            "gt_hands": len(gt_hands),
            "gt_left": any(not h[1] for h in gt_hands),
            "gt_right": any(h[1] for h in gt_hands),
            "models": {},
        }

        # Per-model overlays (stored for side-by-side composition)
        model_overlays = {}
        model_colors = {"wilor": (0, 0, 255), "hamer": (255, 0, 0)}  # Red / Blue in BGR

        for model_name in models_to_run:
            model_result = {"detected": [], "n_detections": 0}

            if model_name == "wilor":
                if args.mode == "detect":
                    detections = run_wilor_detect(wilor_pipe, image_rgb)
                else:
                    if gt_hands:
                        bboxes = [pad_bbox(h[0], img_h, img_w, 1.5) for h in gt_hands]
                        is_rights = [h[1] for h in gt_hands]
                        detections = run_wilor_crop(wilor_pipe, image_rgb, bboxes, is_rights)
                    else:
                        detections = []

            elif model_name == "hamer":
                if gt_hands:
                    bboxes = [pad_bbox(h[0], img_h, img_w, 1.5) for h in gt_hands]
                    is_rights = [h[1] for h in gt_hands]
                    detections = run_hamer_crop(
                        hamer_model, hamer_cfg, image_bgr, bboxes, is_rights, hamer_device
                    )
                else:
                    detections = []

            model_result["n_detections"] = len(detections)
            for det in detections:
                model_result["detected"].append({
                    "is_right": det["is_right"],
                    "keypoints_2d": det["keypoints_2d"].tolist(),
                })

            frame_result["models"][model_name] = model_result

            # Build per-model overlay: GT bboxes + model skeleton
            if not args.no_overlays:
                panel = image_bgr.copy()
                # Draw GT bounding boxes (yellow)
                for bbox_xyxy, is_right in gt_hands:
                    label = "GT R" if is_right else "GT L"
                    draw_bbox(panel, bbox_xyxy, color=(0, 255, 255), thickness=2, label=label)
                # Draw model predictions
                color = model_colors.get(model_name, (0, 255, 0))
                for det in detections:
                    draw_skeleton_2d(panel, det["keypoints_2d"], color=color, thickness=2, radius=3)
                    wrist = det["keypoints_2d"][0]
                    side_label = "R" if det["is_right"] else "L"
                    cv2.putText(panel, side_label,
                                (int(wrist[0]) + 5, int(wrist[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                # Add model title at top
                title = model_name.upper()
                cv2.putText(panel, title, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
                model_overlays[model_name] = panel

        all_results.append(frame_result)

        # Save overlays
        if not args.no_overlays:
            if len(model_overlays) >= 2 and "wilor" in model_overlays and "hamer" in model_overlays:
                # Side-by-side: WiLoR (left) | HaMeR (right)
                combined = np.concatenate(
                    [model_overlays["wilor"], model_overlays["hamer"]], axis=1
                )
                overlay_path = overlay_dir / f"{fid}_overlay.jpg"
                cv2.imwrite(str(overlay_path), combined)
            else:
                # Single model, save as-is
                for mname, panel in model_overlays.items():
                    overlay_path = overlay_dir / f"{fid}_{mname}_overlay.jpg"
                    cv2.imwrite(str(overlay_path), panel)

    # ── Aggregate results ──────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"EGOSURGERY EVALUATION RESULTS")
    print(f"Mode: {args.mode} | Models: {', '.join(models_to_run)}")
    print(f"{'=' * 60}")
    print(f"Total frames evaluated: {len(all_results)}")
    print(f"Time: {elapsed:.1f}s ({len(all_results)/elapsed:.1f} frames/s)")

    # Detection rate summary
    for model_name in models_to_run:
        n_with_detections = 0
        n_total_detections = 0
        n_bimanual = 0
        by_phase = defaultdict(lambda: {"total": 0, "detected": 0})
        by_video = defaultdict(lambda: {"total": 0, "detected": 0})

        for r in all_results:
            if "models" not in r:
                continue
            mr = r["models"].get(model_name, {})
            n_det = mr.get("n_detections", 0)
            if n_det > 0:
                n_with_detections += 1
            if n_det >= 2:
                n_bimanual += 1
            n_total_detections += n_det

            by_phase[r["phase"]]["total"] += 1
            if n_det > 0:
                by_phase[r["phase"]]["detected"] += 1

            by_video[r["video_id"]]["total"] += 1
            if n_det > 0:
                by_video[r["video_id"]]["detected"] += 1

        total = len([r for r in all_results if "models" in r])
        print(f"\n--- {model_name.upper()} ---")
        print(f"  Frames with any detection: {n_with_detections}/{total} "
              f"({100*n_with_detections/total:.1f}%)")
        print(f"  Frames with bimanual:      {n_bimanual}/{total} "
              f"({100*n_bimanual/total:.1f}%)")
        print(f"  Total hand detections:     {n_total_detections}")

        print(f"\n  Detection rate by phase:")
        for phase in sorted(by_phase):
            p = by_phase[phase]
            rate = 100 * p["detected"] / p["total"] if p["total"] > 0 else 0
            print(f"    {phase:<12}: {p['detected']}/{p['total']} ({rate:.0f}%)")

        print(f"\n  Detection rate by video (top/bottom 5):")
        vid_rates = []
        for vid in sorted(by_video):
            v = by_video[vid]
            rate = 100 * v["detected"] / v["total"] if v["total"] > 0 else 0
            vid_rates.append((vid, rate, v["detected"], v["total"]))
        vid_rates.sort(key=lambda x: x[1], reverse=True)
        for vid, rate, det, tot in vid_rates[:5]:
            print(f"    Video {vid}: {det}/{tot} ({rate:.0f}%)")
        if len(vid_rates) > 5:
            print("    ...")
            for vid, rate, det, tot in vid_rates[-3:]:
                print(f"    Video {vid}: {det}/{tot} ({rate:.0f}%)")

    # Save results JSON
    results_path = output_dir / f"egosurgery_{args.mode}_{'_'.join(models_to_run)}_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "mode": args.mode,
            "models": models_to_run,
            "dataset": "EgoSurgery-Phase (bimanual windows)",
            "n_frames": len(all_results),
            "time_seconds": elapsed,
            "phase_distribution": dict(phase_counts),
            "per_frame": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")
    if not args.no_overlays:
        print(f"Overlays saved to {overlay_dir}/")


if __name__ == "__main__":
    main()
