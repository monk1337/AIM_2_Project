#!/usr/bin/env python3
"""Evaluate HaMeR (off-the-shelf) on POV-Surgery frames.

Uses GT-derived bounding boxes (crop-regress mode only, HaMeR has no built-in
detector). Follows the same GT pipeline, metrics, and output format as
run_eval_wilor.py for direct comparison.

Usage:
    python run_eval_hamer.py --mode crop --data-dir ../pov_surgery_data
    python run_eval_hamer.py --mode crop --max-frames 10 --data-dir ../pov_surgery_data
    python run_eval_hamer.py --mode crop --data-dir ../pov_surgery_data --output-dir results/
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
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

# MANO-to-OpenPose joint reordering (same for WiLoR and HaMeR)
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
        flat_hand_mean=True,
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


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate HaMeR on POV-Surgery")
    parser.add_argument("--mode", choices=["crop"], default="crop",
                        help="crop: GT-derived bboxes (HaMeR has no built-in detector)")
    parser.add_argument("--split", type=str, default="demo",
                        help="'demo' (s_scalpel_3 only, default), "
                             "'full' (all test sequences via official split), "
                             "or a sequence name (e.g. 'm_diskplacer_1')")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames (0 = all)")
    parser.add_argument("--data-dir", type=str, default="../pov_surgery_data",
                        help="Path to pov_surgery_data/ (default: ../pov_surgery_data)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto based on split)")
    parser.add_argument("--device", type=str, default=None, help="Device (mps/cuda/cpu)")
    args = parser.parse_args()

    # Resolve paths
    data_dir = Path(args.data_dir).resolve()

    # Determine data root and sequences based on split
    if args.split == "demo":
        # Original behavior: demo_data/POV_Surgery_data/s_scalpel_3
        data_root = data_dir / "demo_data" / "POV_Surgery_data"
        sequences = ["s_scalpel_3"]
        split_label = "s_scalpel_3 (demo)"
        test_info = None  # no pickle for demo
    elif args.split == "full":
        # Full test set via official split pickle
        data_root = data_dir / "demo_data" / "POV_Surgery_data"
        test_pkl = data_root / "handoccnet_train" / "2d_repro_ho3d_style_test_cleaned.pkl"
        if not test_pkl.exists():
            raise FileNotFoundError(
            f"Test split pickle not found at {test_pkl}\n"
            f"Download POV_Surgery_data from: "
            f"https://drive.google.com/drive/folders/1nSDig2cEHscCPgG10-VcSW3Q1zKge4tP"
            )
        with open(test_pkl, "rb") as f:
            test_info = pickle.load(f)
        sequences = sorted(set(k.split("/")[0] for k in test_info.keys()))
        split_label = f"full test set ({len(sequences)} sequences)"
    else:
        # Single sequence by name
        data_root = data_dir / "demo_data" / "POV_Surgery_data"
        sequences = [args.split]
        split_label = args.split
        test_info = None

    mano_model_dir = data_dir / "data" / "bodymodel"

    # Auto output dir
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = Path("results").resolve() / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if torch.backends.mps.is_available():
            print("Note: Using CPU (HaMeR requires float64 ops unsupported by MPS)")
    print(f"Device: {device}")
    print(f"Split: {split_label}")

    # Collect frames. When test_info pickle is available, use it (has pre-stored 2D joints).
    # Otherwise, glob annotation directories.
    annotation_files = []  # list of (key, pkl_path, img_path, joints_uv_or_None)
    if test_info is not None:
        # Use test pickle: iterate its keys, which gives pre-stored 2D joints
        for pkl_key, val in test_info.items():
            seq, frame_id = pkl_key.split("/")
            pkl_path = data_root / "annotation" / seq / f"{frame_id}.pkl"
            img_path = data_root / "color" / seq / f"{frame_id}.jpg"
            if pkl_path.exists() and img_path.exists():
                # Swap axes: pickle stores (y, x) -> we need (x, y)
                juv_raw = val["joints_uv"]
                juv = np.zeros_like(juv_raw)
                juv[:, 0] = juv_raw[:, 1]
                juv[:, 1] = juv_raw[:, 0]
                # Reorder from MANO to OpenPose (matches 3D joint order in this script)
                juv = juv[MANO_TO_OPENPOSE]
                annotation_files.append((pkl_key, pkl_path, img_path, juv))
    else:
        # Glob directories (demo / single sequence mode)
        for seq in sequences:
            ann_dir = data_root / "annotation" / seq
            img_dir = data_root / "color" / seq
            if not ann_dir.exists():
                print(f"  WARNING: annotation dir not found: {ann_dir}")
                continue
            for pkl_path in sorted(ann_dir.glob("*.pkl")):
                frame_id = pkl_path.stem
                img_path = img_dir / f"{frame_id}.jpg"
                if img_path.exists():
                    annotation_files.append((f"{seq}/{frame_id}", pkl_path, img_path, None))

    if not annotation_files:
        print(f"ERROR: No annotation/image pairs found for split '{args.split}'")
        sys.exit(1)
    print(f"Found {len(annotation_files)} frames across {len(sequences)} sequence(s)")

    if args.max_frames > 0:
        annotation_files = annotation_files[: args.max_frames]
        print(f"Processing first {len(annotation_files)} frames")

    # Init output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Init MANO model for GT
    print("Loading MANO model for GT...")
    gt_mano, tip_ids = build_gt_mano(mano_model_dir, device)

    # Init HaMeR model
    print("Loading HaMeR model...")

    # HaMeR uses relative paths from CACHE_DIR_HAMER="./_DATA".
    # Temporarily chdir to project root so these resolve correctly.
    import os
    project_root = Path(__file__).resolve().parent.parent
    old_cwd = os.getcwd()
    os.chdir(project_root)

    import hamer.configs
    hamer.configs.CACHE_DIR_HAMER = str(project_root / "_DATA")

    from hamer.models import load_hamer
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.utils import recursive_to

    hamer_ckpt = project_root / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
    model, model_cfg = load_hamer(str(hamer_ckpt))

    os.chdir(old_cwd)
    model = model.to(device)
    model.eval()
    print("HaMeR loaded.")

    # ── Precompute GT (saved to disk, keyed by seq/frame_id) ─────────
    gt_cache_path = output_dir / f"gt_cache_{args.split}.pkl"
    if gt_cache_path.exists():
        print(f"\nLoading cached GT from {gt_cache_path}...")
        import pickle as pkl_mod
        with open(gt_cache_path, "rb") as f:
            gt_cache = pkl_mod.load(f)
        print(f"  Loaded GT cache: {len(gt_cache)} frames.")
    else:
        print("\nPrecomputing GT for all frames...")
        gt_cache = {}
        for key, pkl_path, img_path, joints_uv in annotation_files:
            gt = load_gt_annotation(pkl_path)
            gt_joints_local, gt_verts_local = mano_forward(
                gt_mano, tip_ids,
                gt["global_orient"], gt["hand_pose"], gt["betas"],
                device,
            )
            gt_joints, gt_verts = transform_to_camera(gt_joints_local, gt_verts_local, gt)

            # Bbox: use pre-stored 2D joints if available (always valid),
            # otherwise fall back to 3D projection (can fail)
            if joints_uv is not None:
                x_min, y_min = joints_uv.min(axis=0)
                x_max, y_max = joints_uv.max(axis=0)
                cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
                s = max(x_max - x_min, y_max - y_min) * 1.5
                bbox = np.array([max(0, cx - s/2), max(0, cy - s/2),
                                 min(1920, cx + s/2), min(1080, cy + s/2)])
            else:
                bbox = derive_bbox_from_gt(gt_joints, K, (1080, 1920), pad_factor=1.5)

            gt_cache[key] = {
                "joints": gt_joints, "verts": gt_verts, "bbox": bbox,
                "joints_2d": joints_uv,  # store for P2D GT
            }
        import pickle as pkl_mod
        with open(gt_cache_path, "wb") as f:
            pkl_mod.dump(gt_cache, f)
        print(f"  GT precomputed and saved: {len(gt_cache)} frames -> {gt_cache_path}")
    bbox_failures = sum(1 for v in gt_cache.values() if v["bbox"] is None)

    # ── Evaluation loop ────────────────────────────────────────────────
    print(f"\nMode: {args.mode}")
    all_metrics = []
    inference_failures = 0
    t_start = time.time()

    for i, (key, pkl_path, img_path, _joints_uv) in enumerate(annotation_files):

        if (i + 1) % 200 == 0 or i == 0:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(annotation_files)}]  {fps:.1f} frames/s  "
                  f"bbox_fail={bbox_failures}  inf_fail={inference_failures}")

        cached = gt_cache[key]
        gt_joints = cached["joints"]
        gt_verts = cached["verts"]
        bbox = cached["bbox"]

        if bbox is None:
            bbox_failures += 1
            all_metrics.append({"frame_id": key, "detected": False, "reason": "bbox_fail"})
            continue

        # Load image as BGR (cv2 convention, what HaMeR expects)
        img_cv2 = cv2.imread(str(img_path))

        # Prepare HaMeR input via ViTDetDataset (index directly, no DataLoader overhead)
        import io
        from contextlib import redirect_stdout
        boxes = np.array([bbox])
        right = np.array([1.0])  # right hand
        with redirect_stdout(io.StringIO()):
            dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=2.5)
            item = dataset[0]

        try:
            batch = {}
            for k, v in item.items():
                if isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(v, device=device).unsqueeze(0).float()
                elif isinstance(v, torch.Tensor):
                    batch[k] = v.unsqueeze(0).to(device)
                elif isinstance(v, (int, float)):
                    batch[k] = torch.tensor([v], device=device).float()
                else:
                    batch[k] = v
            with torch.no_grad():
                out = model(batch)
        except Exception as e:
            inference_failures += 1
            all_metrics.append({"frame_id": key, "detected": False, "reason": f"inference_error: {e}"})
            continue

        # Extract predictions
        pred_joints = out["pred_keypoints_3d"][0].cpu().numpy()  # (21, 3)
        pred_verts = out["pred_vertices"][0].cpu().numpy()  # (778, 3)

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
        # Use HaMeR's native 2D output (pred_keypoints_2d), mapped to full image
        # via box_center/box_size. This is simpler and never produces NaN.
        gt_2d_stored = cached.get("joints_2d")
        if gt_2d_stored is not None:
            gt_2d = gt_2d_stored  # pre-stored from test pickle (always valid)
        else:
            gt_2d_proj = (K @ gt_joints.T).T
            gt_2d = gt_2d_proj[:, :2] / gt_2d_proj[:, 2:3]  # fallback: project from 3D

        pred_2d_norm = out["pred_keypoints_2d"][0].cpu().numpy()  # (21, 2) normalized
        box_center = batch["box_center"][0].cpu().numpy()
        box_size = batch["box_size"].item()
        pred_2d = pred_2d_norm * box_size + box_center  # map to full image pixels
        p2d_error = np.sqrt(((pred_2d - gt_2d) ** 2).sum(axis=-1)).mean()

        metrics = {
            "frame_id": key,
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

    print("\n" + "=" * 60)
    print(f"RESULTS: HaMeR (off-the-shelf, GT-bbox crop-regress)")
    print(f"Dataset: POV-Surgery {split_label}")
    print("=" * 60)
    print(f"Total frames:       {total}")
    print(f"Detected:           {n_det} ({100*n_det/total:.1f}%)")
    print(f"Inference failures: {inference_failures}")
    print(f"Bbox failures:      {bbox_failures}")
    print(f"Time:               {elapsed:.1f}s ({total/elapsed:.1f} frames/s)")

    if n_det > 0:
        mpjpe_vals = [m["mpjpe"] for m in detected]
        pa_mpjpe_vals = [m["pa_mpjpe"] for m in detected]
        pve_vals = [m["pve"] for m in detected]
        pa_pve_vals = [m["pa_pve"] for m in detected]
        p2d_vals = [m["p2d"] for m in detected if not np.isnan(m["p2d"])]
        p2d_skipped = sum(1 for m in detected if np.isnan(m["p2d"]))

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
        if p2d_skipped > 0:
            print(f"  (P2d: {p2d_skipped} frames skipped due to bad camera projection)")

        # Per-finger
        print(f"\n{'Finger':<12} {'MPJPE (mm)':>12}")
        print("-" * 26)
        for fn in ["thumb", "index", "middle", "ring", "pinky"]:
            vals = [m["per_finger"][fn] for m in detected]
            print(f"{fn:<12} {np.mean(vals):>12.2f}")

        print(f"\n{'HaMeR (ours, raw)':<20} MPJPE={np.mean(mpjpe_vals):.2f}  PA-MPJPE={np.mean(pa_mpjpe_vals):.2f}")

    # Save results
    results_path = output_dir / f"hamer_{args.mode}_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": f"HaMeR (off-the-shelf, {args.mode})",
            "dataset": f"POV-Surgery {split_label}",
            "total_frames": total,
            "detected": n_det,
            "inference_failures": inference_failures,
            "detection_failures": 0,
            "time_seconds": elapsed,
            "metrics_summary": {
                "mpjpe_mean": float(np.mean(mpjpe_vals)) if n_det else None,
                "pa_mpjpe_mean": float(np.mean(pa_mpjpe_vals)) if n_det else None,
                "pve_mean": float(np.mean(pve_vals)) if n_det else None,
                "pa_pve_mean": float(np.mean(pa_pve_vals)) if n_det else None,
                "p2d_mean": float(np.mean(p2d_vals)) if p2d_vals else None,
            },
            "per_frame": all_metrics,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
