#!/usr/bin/env python3
"""Evaluate WiLoR (off-the-shelf) on POV-Surgery frames.

Phase 1 of course project: run WiLoR on POV-Surgery demo frames,
compare predicted MANO output against ground-truth annotations,
compute MPJPE, PA-MPJPE, PVE, PA-PVE.

Two modes:
  --mode detect: Use WiLoR's built-in YOLO detector (tests detection + regression)
  --mode crop:   Derive bboxes from GT MANO projection, use predict_with_bboxes (tests regression only)

Usage:
    .venv/bin/python scripts/eval_wilor_pov_surgery.py --mode crop
    .venv/bin/python scripts/eval_wilor_pov_surgery.py --mode detect --max-frames 100
    .venv/bin/python scripts/eval_wilor_pov_surgery.py --mode crop --visualize --max-frames 10
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "pov_surgery_data" / "demo_data" / "POV_Surgery_data"
ANNOTATION_DIR = DEMO_DIR / "annotation" / "s_scalpel_3"
IMAGE_DIR = DEMO_DIR / "color" / "s_scalpel_3"
MANO_MODEL_DIR = PROJECT_ROOT / "pov_surgery_data" / "data" / "bodymodel"
OUTPUT_DIR = PROJECT_ROOT / "pipeline_output" / "eval_pov_surgery"

# POV-Surgery camera intrinsics (fixed across all frames)
K = np.array([
    [1198.4395, 0.0, 960.0],
    [0.0, 1198.4395, 175.2],
    [0.0, 0.0, 1.0],
])

# OpenGL → OpenCV coordinate flip
COORD_CHANGE = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])

# WiLoR's MANO-to-OpenPose joint reordering
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


# ── MANO Ground-Truth Model ───────────────────────────────────────────
def build_gt_mano(device):
    """Build a MANO model matching WiLoR's joint convention (21 joints, OpenPose order)."""
    import smplx
    from smplx.vertex_ids import vertex_ids as smplx_vertex_ids

    mano = smplx.create(
        str(MANO_MODEL_DIR),
        model_type="mano",
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=False,
    ).to(device)

    # Fingertip vertex indices (same as WiLoR uses)
    tip_ids = [
        smplx_vertex_ids["mano"]["thumb"],   # 744
        smplx_vertex_ids["mano"]["index"],   # 320
        smplx_vertex_ids["mano"]["middle"],  # 443
        smplx_vertex_ids["mano"]["ring"],    # 554
        smplx_vertex_ids["mano"]["pinky"],   # 671
    ]

    return mano, tip_ids


def mano_forward(mano_model, tip_ids, global_orient, hand_pose, betas, device):
    """Run MANO FK WITHOUT translation, return 21 joints (OpenPose order) and 778 vertices.

    Returns joints/verts in MANO's local frame (global_orient applied, no translation).
    The caller must apply coordinate transforms separately.
    """
    with torch.no_grad():
        out = mano_model(
            global_orient=torch.tensor(global_orient, dtype=torch.float32, device=device),
            hand_pose=torch.tensor(hand_pose, dtype=torch.float32, device=device),
            betas=torch.tensor(betas, dtype=torch.float32, device=device),
            transl=torch.zeros(1, 3, dtype=torch.float32, device=device),
        )
    joints_16 = out.joints[0].cpu().numpy()      # (16, 3)
    vertices = out.vertices[0].cpu().numpy()      # (778, 3)

    # Add 5 fingertips from vertices
    tips = vertices[tip_ids]                       # (5, 3)
    joints_21 = np.concatenate([joints_16, tips])  # (21, 3)

    # Reorder to OpenPose convention (matching WiLoR output)
    joints_21 = joints_21[MANO_TO_OPENPOSE]        # (21, 3)

    return joints_21, vertices


def transform_to_camera(joints, verts, anno):
    """Transform MANO joints/verts from local frame to OpenCV camera frame.

    Follows POV-Surgery's coordinate transform chain:
    1. grab2world rotation
    2. camera inverse (world→camera)
    3. OpenGL→OpenCV flip
    """
    cam_rot = anno["cam_rot"]
    cam_transl = anno["cam_transl"]
    g2w_R = anno.get("grab2world_R")
    g2w_T = anno.get("grab2world_T")
    transl = anno["transl"]  # MANO translation

    # Build camera pose and inverse
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = cam_transl
    camera_pose[:3, :3] = cam_rot
    cam_inv = np.linalg.inv(camera_pose)

    # Combined rotation: MANO→world→camera
    all_addition_g = g2w_R @ cam_inv[:3, :3].T
    # Combined translation (includes MANO transl)
    all_addition_t = g2w_T @ cam_inv[:3, :3].T + cam_inv[:3, 3]
    all_addition_t_no_transl = transl @ all_addition_g + all_addition_t

    # Apply to joints and verts
    joints_cam = joints @ all_addition_g + all_addition_t_no_transl
    verts_cam = verts @ all_addition_g + all_addition_t_no_transl

    # OpenGL → OpenCV (flip Y and Z)
    joints_cv = joints_cam @ COORD_CHANGE.T
    verts_cv = verts_cam @ COORD_CHANGE.T

    return joints_cv, verts_cv


# ── Metrics ────────────────────────────────────────────────────────────
def procrustes_align(S1, S2):
    """Align S1 to S2 using similarity transform (rotation + scale + translation).

    Args:
        S1: (N, 3) predicted points
        S2: (N, 3) ground-truth points
    Returns:
        S1_hat: (N, 3) aligned predicted points
    """
    mu1 = S1.mean(axis=0, keepdims=True)
    mu2 = S2.mean(axis=0, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = np.sum(X1 ** 2)
    if var1 < 1e-10:
        return S1  # degenerate case

    K_mat = X1.T @ X2  # (3, 3)
    U, s, Vt = np.linalg.svd(K_mat)
    Z = np.eye(3)
    Z[-1, -1] *= np.sign(np.linalg.det(U @ Vt))
    R = Vt.T @ Z @ U.T
    scale = np.trace(R @ K_mat) / var1
    t = mu2 - scale * (R @ mu1.T).T

    S1_hat = scale * (S1 @ R.T) + t
    return S1_hat


def compute_mpjpe(pred, gt):
    """Mean per-joint position error in mm."""
    return np.sqrt(((pred - gt) ** 2).sum(axis=-1)).mean() * 1000


def compute_pa_mpjpe(pred, gt):
    """PA-MPJPE: Procrustes-aligned MPJPE in mm."""
    pred_aligned = procrustes_align(pred, gt)
    return np.sqrt(((pred_aligned - gt) ** 2).sum(axis=-1)).mean() * 1000


def compute_per_finger_mpjpe(pred, gt):
    """Per-finger MPJPE (OpenPose ordering: thumb=1-4, index=5-8, etc.)."""
    # Root-relative
    pred_r = pred - pred[0:1]
    gt_r = gt - gt[0:1]

    fingers = {
        "thumb": slice(1, 5),
        "index": slice(5, 9),
        "middle": slice(9, 13),
        "ring": slice(13, 17),
        "pinky": slice(17, 21),
    }
    result = {}
    for name, sl in fingers.items():
        err = np.sqrt(((pred_r[sl] - gt_r[sl]) ** 2).sum(axis=-1)).mean() * 1000
        result[name] = err
    return result


# ── GT Loading ─────────────────────────────────────────────────────────
def load_gt_annotation(pkl_path):
    """Load POV-Surgery ground-truth annotation."""
    with open(pkl_path, "rb") as f:
        anno = pickle.load(f)

    mano_params = anno["mano"]
    return {
        "global_orient": np.array(mano_params["global_orient"], dtype=np.float32),  # (1, 3)
        "hand_pose": np.array(mano_params["hand_pose"], dtype=np.float32),          # (1, 45)
        "betas": np.array(mano_params["betas"], dtype=np.float32),                  # (1, 10)
        "transl": np.array(mano_params["transl"], dtype=np.float32),                # (1, 3)
        "cam_rot": np.array(anno.get("cam_rot", np.eye(3)), dtype=np.float64),
        "cam_transl": np.array(anno.get("cam_transl", np.zeros(3)), dtype=np.float64),
        "grab2world_R": np.array(anno.get("grab2world_R", np.eye(3)), dtype=np.float64),
        "grab2world_T": np.array(anno.get("grab2world_T", np.zeros((1, 3))), dtype=np.float64),
    }


# ── Bbox from GT ───────────────────────────────────────────────────────
def derive_bbox_from_gt(gt_joints_21, K, img_shape, pad_factor=1.5):
    """Project GT 3D joints to 2D and derive a bounding box with padding.

    Args:
        gt_joints_21: (21, 3) 3D joints
        K: (3, 3) camera intrinsics
        img_shape: (H, W)
        pad_factor: bbox expansion factor (1.5 = 50% padding on each side)
    Returns:
        bbox_xyxy: [x1, y1, x2, y2] or None if projection fails
    """
    # Project to 2D
    pts_2d = (K @ gt_joints_21.T).T  # (21, 3)
    z = pts_2d[:, 2]
    valid = z > 0.01
    if valid.sum() < 5:
        return None

    pts_2d = pts_2d[valid, :2] / pts_2d[valid, 2:3]  # (N, 2)

    x_min, y_min = pts_2d.min(axis=0)
    x_max, y_max = pts_2d.max(axis=0)

    # Pad
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = (x_max - x_min) * pad_factor
    h = (y_max - y_min) * pad_factor
    s = max(w, h)  # square bbox

    x1 = max(0, cx - s / 2)
    y1 = max(0, cy - s / 2)
    x2 = min(img_shape[1], cx + s / 2)
    y2 = min(img_shape[0], cy + s / 2)

    return np.array([x1, y1, x2, y2])


# ── WiLoR Prediction ──────────────────────────────────────────────────
def extract_right_hand(outputs):
    """From WiLoR outputs, find the right hand detection."""
    for out in outputs:
        if out["is_right"] == 1.0:
            preds = out["wilor_preds"]
            return {
                "joints_3d": preds["pred_keypoints_3d"][0].copy(),    # (21, 3)
                "vertices": preds["pred_vertices"][0].copy(),          # (778, 3)
                "global_orient": preds["global_orient"][0].copy(),     # (1, 3) or (3,)
                "hand_pose": preds["hand_pose"][0].copy(),             # (15, 3)
                "betas": preds["betas"][0].copy(),                     # (10,)
            }
    return None


def extract_first_hand(outputs):
    """From WiLoR crop-regress outputs, extract the first (only) result."""
    if len(outputs) == 0:
        return None
    preds = outputs[0]["wilor_preds"]
    return {
        "joints_3d": preds["pred_keypoints_3d"][0].copy(),
        "vertices": preds["pred_vertices"][0].copy(),
        "global_orient": preds["global_orient"][0].copy(),
        "hand_pose": preds["hand_pose"][0].copy(),
        "betas": preds["betas"][0].copy(),
    }


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate WiLoR on POV-Surgery")
    parser.add_argument("--mode", choices=["detect", "crop"], default="crop",
                        help="detect: use YOLO detector; crop: GT-derived bboxes (default: crop)")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
    parser.add_argument("--visualize", action="store_true", help="Save overlay visualizations")
    parser.add_argument("--device", type=str, default=None, help="Device (mps/cuda/cpu)")
    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    # Check data exists
    annotation_files = sorted(ANNOTATION_DIR.glob("*.pkl"))
    image_files = sorted(IMAGE_DIR.glob("*.jpg"))
    if not annotation_files:
        print(f"ERROR: No annotations found in {ANNOTATION_DIR}")
        sys.exit(1)
    print(f"Found {len(annotation_files)} annotations, {len(image_files)} images")

    if args.max_frames > 0:
        annotation_files = annotation_files[: args.max_frames]
        print(f"Processing first {len(annotation_files)} frames")

    # Init output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        (OUTPUT_DIR / "vis").mkdir(exist_ok=True)

    # Init MANO model for GT forward kinematics
    print("Loading MANO model for GT...")
    gt_mano, tip_ids = build_gt_mano(device)

    # Init WiLoR pipeline
    print("Loading WiLoR pipeline...")

    # Monkey-patch torch.load for compatibility (same as step2 scripts)
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
    print("WiLoR loaded.")

    # ── Evaluation loop ────────────────────────────────────────────────
    print(f"\nMode: {args.mode}")
    all_metrics = []
    detection_failures = 0
    bbox_failures = 0
    t_start = time.time()

    for i, pkl_path in enumerate(annotation_files):
        frame_id = pkl_path.stem  # e.g. "00001"
        img_path = IMAGE_DIR / f"{frame_id}.jpg"

        if not img_path.exists():
            continue

        # Progress
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(annotation_files)}]  {fps:.1f} frames/s  "
                  f"det_fail={detection_failures}  bbox_fail={bbox_failures}")

        # Load GT
        gt = load_gt_annotation(pkl_path)

        # GT MANO forward kinematics (no translation) → 21 joints, 778 vertices
        gt_joints_local, gt_verts_local = mano_forward(
            gt_mano, tip_ids,
            gt["global_orient"], gt["hand_pose"], gt["betas"],
            device,
        )

        # Transform to OpenCV camera frame for 2D projection
        gt_joints, gt_verts = transform_to_camera(gt_joints_local, gt_verts_local, gt)

        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))

        # Run WiLoR
        if args.mode == "detect":
            outputs = pipe.predict(image)
            pred = extract_right_hand(outputs)
        else:
            # Crop-regress mode: derive bbox from GT MANO 2D projection
            bbox = derive_bbox_from_gt(gt_joints, K, image.shape[:2], pad_factor=1.5)
            if bbox is None:
                bbox_failures += 1
                all_metrics.append({"frame_id": frame_id, "detected": False, "reason": "bbox_fail"})
                continue
            bboxes = np.array([bbox])
            is_rights = [1.0]  # right hand
            outputs = pipe.predict_with_bboxes(image, bboxes, is_rights, rescale_factor=2.5)
            pred = extract_first_hand(outputs)

        if pred is None:
            detection_failures += 1
            all_metrics.append({"frame_id": frame_id, "detected": False, "reason": "no_detection"})
            continue

        pred_joints = pred["joints_3d"]   # (21, 3)
        pred_verts = pred["vertices"]     # (778, 3)

        # ── Compute metrics ────────────────────────────────────────────
        # For WiLoR predictions: pred_joints are in camera space but centered differently.
        # For GT: joints come from MANO FK with GT params (in GT coordinate frame).
        # We use root-relative (subtract wrist) and PA alignment to handle frame differences.

        # Root-relative MPJPE
        pred_root_rel = pred_joints - pred_joints[0:1]
        gt_root_rel = gt_joints - gt_joints[0:1]
        mpjpe = np.sqrt(((pred_root_rel - gt_root_rel) ** 2).sum(axis=-1)).mean() * 1000

        # PA-MPJPE
        pa_mpjpe = compute_pa_mpjpe(pred_joints, gt_joints)

        # Root-relative PVE
        pred_verts_rel = pred_verts - pred_joints[0:1]
        gt_verts_rel = gt_verts - gt_joints[0:1]  # use joint[0] for consistency
        pve = np.sqrt(((pred_verts_rel - gt_verts_rel) ** 2).sum(axis=-1)).mean() * 1000

        # PA-PVE
        pred_verts_aligned = procrustes_align(pred_verts, gt_verts)
        pa_pve = np.sqrt(((pred_verts_aligned - gt_verts) ** 2).sum(axis=-1)).mean() * 1000

        # Per-finger breakdown
        per_finger = compute_per_finger_mpjpe(pred_joints, gt_joints)

        # 2D reprojection error: use WiLoR's predicted 2D keypoints vs GT projected to 2D
        gt_2d = (K @ gt_joints.T).T
        gt_2d = gt_2d[:, :2] / gt_2d[:, 2:3]
        # Use WiLoR's own 2D keypoints (already in image space)
        if args.mode == "crop":
            wilor_kp2d = outputs[0]["wilor_preds"].get("pred_keypoints_2d")
        else:
            # In detect mode, find the right hand output
            wilor_kp2d = None
            for out in outputs:
                if out["is_right"] == 1.0:
                    wilor_kp2d = out["wilor_preds"].get("pred_keypoints_2d")
                    break
        if wilor_kp2d is not None:
            pred_2d = wilor_kp2d[0]  # (21, 2)
            p2d_error = np.sqrt(((pred_2d - gt_2d) ** 2).sum(axis=-1)).mean()
        else:
            p2d_error = float("nan")

        metrics = {
            "frame_id": frame_id,
            "detected": True,
            "mpjpe": float(mpjpe),
            "pa_mpjpe": float(pa_mpjpe),
            "pve": float(pve),
            "pa_pve": float(pa_pve),
            "p2d": float(p2d_error),
            "per_finger": {k: float(v) for k, v in per_finger.items()},
        }
        all_metrics.append(metrics)

        # Visualization
        if args.visualize:
            try:
                import cv2
                vis = image.copy()
                # Draw GT joints in green
                for j in range(21):
                    if gt_2d[j, 0] > 0 and gt_2d[j, 1] > 0:
                        cv2.circle(vis, (int(gt_2d[j, 0]), int(gt_2d[j, 1])), 3, (0, 255, 0), -1)
                # Draw pred keypoints from WiLoR in red
                pred_kp2d = outputs[0]["wilor_preds"].get("pred_keypoints_2d")
                if pred_kp2d is not None:
                    for j in range(21):
                        x, y = int(pred_kp2d[0, j, 0]), int(pred_kp2d[0, j, 1])
                        cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)
                # Add text
                cv2.putText(vis, f"PA-MPJPE: {pa_mpjpe:.1f}mm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(vis, f"MPJPE: {mpjpe:.1f}mm", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                vis_path = OUTPUT_DIR / "vis" / f"{frame_id}.jpg"
                cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            except ImportError:
                pass  # cv2 not available

    # ── Aggregate results ──────────────────────────────────────────────
    elapsed = time.time() - t_start
    detected = [m for m in all_metrics if m.get("detected", False)]
    total = len(all_metrics)
    n_det = len(detected)

    mode_label = "YOLO detect" if args.mode == "detect" else "GT-bbox crop-regress"
    print("\n" + "=" * 60)
    print(f"RESULTS: WiLoR (off-the-shelf, {mode_label})")
    print(f"Dataset: POV-Surgery s_scalpel_3 (demo)")
    print("=" * 60)
    print(f"Total frames:       {total}")
    print(f"Detected:           {n_det} ({100*n_det/total:.1f}%)")
    print(f"Detection failures: {detection_failures} ({100*detection_failures/total:.1f}%)")
    if args.mode == "crop":
        print(f"Bbox failures:      {bbox_failures}")
    print(f"Time:               {elapsed:.1f}s ({total/elapsed:.1f} frames/s)")

    if n_det > 0:
        mpjpe_vals = [m["mpjpe"] for m in detected]
        pa_mpjpe_vals = [m["pa_mpjpe"] for m in detected]
        pve_vals = [m["pve"] for m in detected]
        pa_pve_vals = [m["pa_pve"] for m in detected]
        p2d_vals = [m["p2d"] for m in detected]

        print(f"\n{'Metric':<20} {'Mean':>8} {'Median':>8} {'Std':>8}")
        print("-" * 48)
        for name, vals in [
            ("MPJPE (mm)", mpjpe_vals),
            ("PA-MPJPE (mm)", pa_mpjpe_vals),
            ("PVE (mm)", pve_vals),
            ("PA-PVE (mm)", pa_pve_vals),
            ("P2d (px)", p2d_vals),
        ]:
            arr = np.array(vals)
            print(f"{name:<20} {arr.mean():>8.2f} {np.median(arr):>8.2f} {arr.std():>8.2f}")

        # Per-finger breakdown
        print(f"\n{'Finger':<12} {'MPJPE (mm)':>12}")
        print("-" * 26)
        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        for fn in finger_names:
            vals = [m["per_finger"][fn] for m in detected]
            print(f"{fn:<12} {np.mean(vals):>12.2f}")

        # Comparison with CPCI paper (fine-tuned numbers)
        print("\n--- Reference: CPCI paper (all methods FINE-TUNED on POV-Surgery) ---")
        print(f"{'Method':<20} {'MPJPE':>8} {'PA-MPJPE':>10}")
        print("-" * 40)
        print(f"{'WiLoR (CPCI)': <20} {'13.72':>8} {'4.33':>10}")
        print(f"{'HaMeR (CPCI)':<20} {'13.15':>8} {'4.41':>10}")
        print(f"{'CPCI (SOTA)':<20} {'12.21':>8} {'4.21':>10}")
        print(f"{'WiLoR (ours, raw)':<20} {np.mean(mpjpe_vals):>8.2f} {np.mean(pa_mpjpe_vals):>10.2f}")

    # Save results
    results_path = OUTPUT_DIR / "wilor_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": f"WiLoR (off-the-shelf, {args.mode})",
            "dataset": "POV-Surgery s_scalpel_3 (demo)",
            "total_frames": total,
            "detected": n_det,
            "detection_failures": detection_failures,
            "time_seconds": elapsed,
            "metrics_summary": {
                "mpjpe_mean": float(np.mean(mpjpe_vals)) if n_det else None,
                "pa_mpjpe_mean": float(np.mean(pa_mpjpe_vals)) if n_det else None,
                "pve_mean": float(np.mean(pve_vals)) if n_det else None,
                "pa_pve_mean": float(np.mean(pa_pve_vals)) if n_det else None,
                "p2d_mean": float(np.mean(p2d_vals)) if n_det else None,
            },
            "per_frame": all_metrics,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
