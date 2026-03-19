#!/usr/bin/env python3
"""Evaluate WiLoR (off-the-shelf) on POV-Surgery frames.

Two modes:
  --mode detect: Use WiLoR's built-in YOLO detector (tests detection + regression)
  --mode crop:   Derive bboxes from GT MANO projection, use predict_with_bboxes (regression only)

Usage:
    python run_eval_wilor.py --mode crop --data-dir ../pov_surgery_data
    python run_eval_wilor.py --mode detect --max-frames 100
    python run_eval_wilor.py --mode crop --data-dir ../pov_surgery_data --output-dir results/
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

# POV-Surgery camera intrinsics (fixed across all frames)
K = np.array([
    [1198.4395, 0.0, 960.0],
    [0.0, 1198.4395, 175.2],
    [0.0, 0.0, 1.0],
])

# OpenGL -> OpenCV coordinate flip
COORD_CHANGE = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])

# WiLoR's MANO-to-OpenPose joint reordering
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


# ── MANO Ground-Truth Model ───────────────────────────────────────────
def build_gt_mano(mano_model_dir, device):
    """Build a MANO model matching WiLoR's joint convention (21 joints, OpenPose order)."""
    import smplx
    from smplx.vertex_ids import vertex_ids as smplx_vertex_ids

    mano = smplx.create(
        str(mano_model_dir),
        model_type="mano",
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=False,
    ).to(device)

    tip_ids = [
        smplx_vertex_ids["mano"]["thumb"],
        smplx_vertex_ids["mano"]["index"],
        smplx_vertex_ids["mano"]["middle"],
        smplx_vertex_ids["mano"]["ring"],
        smplx_vertex_ids["mano"]["pinky"],
    ]

    return mano, tip_ids


def mano_forward(mano_model, tip_ids, global_orient, hand_pose, betas, device):
    """Run MANO FK, return 21 joints (OpenPose order) and 778 vertices."""
    with torch.no_grad():
        out = mano_model(
            global_orient=torch.tensor(global_orient, dtype=torch.float32, device=device),
            hand_pose=torch.tensor(hand_pose, dtype=torch.float32, device=device),
            betas=torch.tensor(betas, dtype=torch.float32, device=device),
            transl=torch.zeros(1, 3, dtype=torch.float32, device=device),
        )
    joints_16 = out.joints[0].cpu().numpy()
    vertices = out.vertices[0].cpu().numpy()

    tips = vertices[tip_ids]
    joints_21 = np.concatenate([joints_16, tips])
    joints_21 = joints_21[MANO_TO_OPENPOSE]

    return joints_21, vertices


def transform_to_camera(joints, verts, anno):
    """Transform MANO joints/verts from local frame to OpenCV camera frame."""
    cam_rot = anno["cam_rot"]
    cam_transl = anno["cam_transl"]
    g2w_R = anno.get("grab2world_R")
    g2w_T = anno.get("grab2world_T")
    transl = anno["transl"]

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = cam_transl
    camera_pose[:3, :3] = cam_rot
    cam_inv = np.linalg.inv(camera_pose)

    all_addition_g = g2w_R @ cam_inv[:3, :3].T
    all_addition_t = g2w_T @ cam_inv[:3, :3].T + cam_inv[:3, 3]
    all_addition_t_no_transl = transl @ all_addition_g + all_addition_t

    joints_cam = joints @ all_addition_g + all_addition_t_no_transl
    verts_cam = verts @ all_addition_g + all_addition_t_no_transl

    joints_cv = joints_cam @ COORD_CHANGE.T
    verts_cv = verts_cam @ COORD_CHANGE.T

    return joints_cv, verts_cv


# ── Metrics ────────────────────────────────────────────────────────────
def procrustes_align(S1, S2):
    """Align S1 to S2 via similarity transform (rotation + scale + translation)."""
    mu1 = S1.mean(axis=0, keepdims=True)
    mu2 = S2.mean(axis=0, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = np.sum(X1 ** 2)
    if var1 < 1e-10:
        return S1

    K_mat = X1.T @ X2
    U, s, Vt = np.linalg.svd(K_mat)
    Z = np.eye(3)
    Z[-1, -1] *= np.sign(np.linalg.det(U @ Vt))
    R = Vt.T @ Z @ U.T
    scale = np.trace(R @ K_mat) / var1
    t = mu2 - scale * (R @ mu1.T).T

    S1_hat = scale * (S1 @ R.T) + t
    return S1_hat


def compute_mpjpe(pred, gt):
    return np.sqrt(((pred - gt) ** 2).sum(axis=-1)).mean() * 1000


def compute_pa_mpjpe(pred, gt):
    pred_aligned = procrustes_align(pred, gt)
    return np.sqrt(((pred_aligned - gt) ** 2).sum(axis=-1)).mean() * 1000


def compute_per_finger_mpjpe(pred, gt):
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
    with open(pkl_path, "rb") as f:
        anno = pickle.load(f)

    mano_params = anno["mano"]
    return {
        "global_orient": np.array(mano_params["global_orient"], dtype=np.float32),
        "hand_pose": np.array(mano_params["hand_pose"], dtype=np.float32),
        "betas": np.array(mano_params["betas"], dtype=np.float32),
        "transl": np.array(mano_params["transl"], dtype=np.float32),
        "cam_rot": np.array(anno.get("cam_rot", np.eye(3)), dtype=np.float64),
        "cam_transl": np.array(anno.get("cam_transl", np.zeros(3)), dtype=np.float64),
        "grab2world_R": np.array(anno.get("grab2world_R", np.eye(3)), dtype=np.float64),
        "grab2world_T": np.array(anno.get("grab2world_T", np.zeros((1, 3))), dtype=np.float64),
    }


# ── Bbox from GT ───────────────────────────────────────────────────────
def derive_bbox_from_gt(gt_joints_21, K, img_shape, pad_factor=1.5):
    pts_2d = (K @ gt_joints_21.T).T
    z = pts_2d[:, 2]
    valid = z > 0.01
    if valid.sum() < 5:
        return None

    pts_2d = pts_2d[valid, :2] / pts_2d[valid, 2:3]

    x_min, y_min = pts_2d.min(axis=0)
    x_max, y_max = pts_2d.max(axis=0)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = (x_max - x_min) * pad_factor
    h = (y_max - y_min) * pad_factor
    s = max(w, h)

    x1 = max(0, cx - s / 2)
    y1 = max(0, cy - s / 2)
    x2 = min(img_shape[1], cx + s / 2)
    y2 = min(img_shape[0], cy + s / 2)

    return np.array([x1, y1, x2, y2])


# ── WiLoR Prediction ──────────────────────────────────────────────────
def extract_right_hand(outputs):
    for out in outputs:
        if out["is_right"] == 1.0:
            preds = out["wilor_preds"]
            return {
                "joints_3d": preds["pred_keypoints_3d"][0].copy(),
                "vertices": preds["pred_vertices"][0].copy(),
                "global_orient": preds["global_orient"][0].copy(),
                "hand_pose": preds["hand_pose"][0].copy(),
                "betas": preds["betas"][0].copy(),
            }
    return None


def extract_first_hand(outputs):
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
                        help="detect: YOLO detector; crop: GT-derived bboxes (default: crop)")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames (0 = all)")
    parser.add_argument("--data-dir", type=str, default="../pov_surgery_data",
                        help="Path to pov_surgery_data/ (default: ../pov_surgery_data)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory (default: results/)")
    parser.add_argument("--device", type=str, default=None, help="Device (mps/cuda/cpu)")
    args = parser.parse_args()

    # Resolve paths
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    demo_dir = data_dir / "demo_data" / "POV_Surgery_data"
    annotation_dir = demo_dir / "annotation" / "s_scalpel_3"
    image_dir = demo_dir / "color" / "s_scalpel_3"
    mano_model_dir = data_dir / "data" / "bodymodel"

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
    annotation_files = sorted(annotation_dir.glob("*.pkl"))
    image_files = sorted(image_dir.glob("*.jpg"))
    if not annotation_files:
        print(f"ERROR: No annotations found in {annotation_dir}")
        sys.exit(1)
    print(f"Found {len(annotation_files)} annotations, {len(image_files)} images")

    if args.max_frames > 0:
        annotation_files = annotation_files[: args.max_frames]
        print(f"Processing first {len(annotation_files)} frames")

    # Init output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Init MANO model for GT
    print("Loading MANO model for GT...")
    gt_mano, tip_ids = build_gt_mano(mano_model_dir, device)

    # Init WiLoR pipeline
    print("Loading WiLoR pipeline...")
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
        frame_id = pkl_path.stem
        img_path = image_dir / f"{frame_id}.jpg"

        if not img_path.exists():
            continue

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(annotation_files)}]  {fps:.1f} frames/s  "
                  f"det_fail={detection_failures}  bbox_fail={bbox_failures}")

        gt = load_gt_annotation(pkl_path)

        gt_joints_local, gt_verts_local = mano_forward(
            gt_mano, tip_ids,
            gt["global_orient"], gt["hand_pose"], gt["betas"],
            device,
        )

        gt_joints, gt_verts = transform_to_camera(gt_joints_local, gt_verts_local, gt)

        image = np.array(Image.open(img_path).convert("RGB"))

        if args.mode == "detect":
            outputs = pipe.predict(image)
            pred = extract_right_hand(outputs)
        else:
            bbox = derive_bbox_from_gt(gt_joints, K, image.shape[:2], pad_factor=1.5)
            if bbox is None:
                bbox_failures += 1
                all_metrics.append({"frame_id": frame_id, "detected": False, "reason": "bbox_fail"})
                continue
            bboxes = np.array([bbox])
            is_rights = [1.0]
            outputs = pipe.predict_with_bboxes(image, bboxes, is_rights, rescale_factor=2.5)
            pred = extract_first_hand(outputs)

        if pred is None:
            detection_failures += 1
            all_metrics.append({"frame_id": frame_id, "detected": False, "reason": "no_detection"})
            continue

        pred_joints = pred["joints_3d"]
        pred_verts = pred["vertices"]

        # Root-relative MPJPE
        pred_root_rel = pred_joints - pred_joints[0:1]
        gt_root_rel = gt_joints - gt_joints[0:1]
        mpjpe = np.sqrt(((pred_root_rel - gt_root_rel) ** 2).sum(axis=-1)).mean() * 1000

        # PA-MPJPE
        pa_mpjpe = compute_pa_mpjpe(pred_joints, gt_joints)

        # Root-relative PVE
        pred_verts_rel = pred_verts - pred_joints[0:1]
        gt_verts_rel = gt_verts - gt_joints[0:1]
        pve = np.sqrt(((pred_verts_rel - gt_verts_rel) ** 2).sum(axis=-1)).mean() * 1000

        # PA-PVE
        pred_verts_aligned = procrustes_align(pred_verts, gt_verts)
        pa_pve = np.sqrt(((pred_verts_aligned - gt_verts) ** 2).sum(axis=-1)).mean() * 1000

        # Per-finger
        per_finger = compute_per_finger_mpjpe(pred_joints, gt_joints)

        # 2D reprojection error
        gt_2d = (K @ gt_joints.T).T
        gt_2d = gt_2d[:, :2] / gt_2d[:, 2:3]
        if args.mode == "crop":
            wilor_kp2d = outputs[0]["wilor_preds"].get("pred_keypoints_2d")
        else:
            wilor_kp2d = None
            for out in outputs:
                if out["is_right"] == 1.0:
                    wilor_kp2d = out["wilor_preds"].get("pred_keypoints_2d")
                    break
        if wilor_kp2d is not None:
            pred_2d = wilor_kp2d[0]
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

        # Per-finger
        print(f"\n{'Finger':<12} {'MPJPE (mm)':>12}")
        print("-" * 26)
        for fn in ["thumb", "index", "middle", "ring", "pinky"]:
            vals = [m["per_finger"][fn] for m in detected]
            print(f"{fn:<12} {np.mean(vals):>12.2f}")

        print(f"\n{'WiLoR (ours, raw)':<20} MPJPE={np.mean(mpjpe_vals):.2f}  PA-MPJPE={np.mean(pa_mpjpe_vals):.2f}")

    # Save results
    results_path = output_dir / f"wilor_{args.mode}_results.json"
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
