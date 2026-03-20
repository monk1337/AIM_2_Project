#!/usr/bin/env python3
"""Evaluate HandOCCNet (off-the-shelf) on POV-Surgery frames.

Uses GT-derived bounding boxes (crop-regress mode). Shares the same GT pipeline,
metrics, and output format as run_eval_hamer.py for direct comparison.

HandOCCNet uses its own preprocessing (affine crop to 256x256 with 1.5x bbox
expansion), matching the original POV-Surgery data loader.

Usage:
    python run_eval_handoccnet.py --mode crop --data-dir ../pov_surgery_data --device cuda
    python run_eval_handoccnet.py --mode crop --data-dir ../pov_surgery_data --split full --device cuda
    python run_eval_handoccnet.py --mode crop --max-frames 10 --data-dir ../pov_surgery_data --device cuda
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# POV-Surgery camera intrinsics (fixed across all frames)
K = np.array([
    [1198.4395, 0.0, 960.0],
    [0.0, 1198.4395, 175.2],
    [0.0, 0.0, 1.0],
])

# OpenGL -> OpenCV coordinate flip
COORD_CHANGE = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])

# MANO-to-OpenPose joint reordering
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


# ── MANO Ground-Truth Model ───────────────────────────────────────────
def build_gt_mano(mano_model_dir, device):
    """Build a MANO model for GT (21 joints, OpenPose order)."""
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


# ── HandOCCNet Model ──────────────────────────────────────────────────
def load_handoccnet(ckpt_path, device):
    """Load HandOCCNet model from checkpoint."""
    HANDOCC_ROOT = str(Path(__file__).resolve().parent.parent.parent / "POV_Surgery" / "HandOccNet_ft")
    sys.path.insert(0, HANDOCC_ROOT)
    sys.path.insert(0, os.path.join(HANDOCC_ROOT, 'main'))
    sys.path.insert(0, os.path.join(HANDOCC_ROOT, 'common'))

    from nets.backbone import FPN
    from nets.transformer import Transformer
    from nets.regressor import Regressor
    from main.model import Model

    backbone = FPN(pretrained=False)
    fIT = Transformer(injection=True)
    sET = Transformer(injection=False)
    regressor = Regressor()
    model = Model(backbone, fIT, sET, regressor)
    model = nn.DataParallel(model).cuda()

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['network'], strict=False)
    model.eval()
    epoch = ckpt.get('epoch', '?')
    print(f"HandOCCNet loaded (epoch {epoch}, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return model


def handoccnet_preprocess(img_bgr, bbox):
    """
    Preprocess image for HandOCCNet: crop and resize to 256x256.
    Uses the same affine transform as POV-Surgery's data_aug_val.

    Args:
        img_bgr: (H, W, 3) BGR image
        bbox: (4,) [x1, y1, x2, y2]
    Returns:
        img_tensor: (1, 3, 256, 256) normalized tensor
    """
    from PIL import Image as PILImage
    inp_res = 256

    # Convert bbox to center/scale format
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(x2 - x1, y2 - y1)

    # Build affine transform (center crop, no rotation)
    src = np.array([
        [cx - s / 2, cy - s / 2],
        [cx + s / 2, cy - s / 2],
        [cx + s / 2, cy + s / 2],
    ], dtype=np.float32)
    dst = np.array([
        [0, 0],
        [inp_res, 0],
        [inp_res, inp_res],
    ], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)

    # Warp image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    warped = cv2.warpAffine(img_rgb, M, (inp_res, inp_res), flags=cv2.INTER_LINEAR)

    # To tensor (0-1, CHW)
    img_tensor = torch.from_numpy(warped).float().permute(2, 0, 1) / 255.0
    return img_tensor.unsqueeze(0), M


def handoccnet_inference(model, img_tensor, anno_data, device):
    """
    Run HandOCCNet inference.

    HandOCCNet's forward pass requires GT targets (for internal GT computation
    and metric calculation). We construct the minimal targets dict needed.

    Returns pred_joints (21, 3) and pred_verts (778, 3), both root-centered
    in OpenPose order.
    """
    # The model's forward in 'my_val' mode computes metrics internally.
    # We need to construct the targets dict that it expects.
    # Key fields: mano_param, all_addition_g, all_addition_t_no_transl,
    #             rot_aug, joints2d, bbox_hand, scale_img

    mano_param = np.concatenate([
        anno_data["global_orient"].flatten(),
        anno_data["hand_pose"].flatten(),
        anno_data["betas"].flatten(),
    ])

    # Camera transforms
    cam_rot = anno_data["cam_rot"]
    cam_transl = anno_data["cam_transl"]
    g2w_R = anno_data["grab2world_R"]
    g2w_T = anno_data["grab2world_T"]
    transl = anno_data["transl"]

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = cam_transl
    camera_pose[:3, :3] = cam_rot
    cam_inv = np.linalg.inv(camera_pose)

    all_addition_g = g2w_R @ cam_inv[:3, :3].T
    all_addition_t = g2w_T @ cam_inv[:3, :3].T + cam_inv[:3, 3]
    all_addition_t_no_transl = transl @ all_addition_g + all_addition_t

    # rot_aug is identity during validation
    rot_aug = np.eye(3, dtype=np.float32)

    # Dummy joints2d and bbox (model needs them but we use our own metrics)
    dummy_joints2d = np.zeros((21, 2), dtype=np.float32)
    dummy_bbox_hand = np.array([0, 0, 256, 256], dtype=np.float32)

    targets = {
        "mano_param": torch.from_numpy(mano_param).float().unsqueeze(0).to(device),
        "all_addition_g": torch.from_numpy(all_addition_g.astype(np.float32)).unsqueeze(0).to(device),
        "all_addition_t_no_transl": torch.from_numpy(all_addition_t_no_transl.astype(np.float32)).unsqueeze(0).to(device),
        "rot_aug": torch.from_numpy(rot_aug).unsqueeze(0).to(device),
        "joints2d": torch.from_numpy(dummy_joints2d).unsqueeze(0).to(device),
        "bbox_hand": torch.from_numpy(dummy_bbox_hand).unsqueeze(0).to(device),
        "scale_img": torch.tensor([256.0]).to(device),
    }
    meta_info = {"root_joint_cam": torch.from_numpy(rot_aug).unsqueeze(0)}
    inputs = {"img": img_tensor.to(device)}

    # Hook into regressor to capture pred_mano_results
    captured = {}
    model_inner = model.module
    orig_forward = model_inner.regressor.forward

    def hooked_forward(feats, gt_mano_params=None):
        res = orig_forward(feats, gt_mano_params)
        captured["pred"] = res[0]  # pred_mano_results
        return res

    model_inner.regressor.forward = hooked_forward
    # Use step not divisible by 10 to skip debug viz
    try:
        with torch.no_grad():
            model(inputs, targets, meta_info, 'my_val', this_name='000001')
    finally:
        model_inner.regressor.forward = orig_forward

    pred_joints = captured["pred"]["joints3d"][0].cpu().numpy()  # (21, 3) root-centered
    pred_verts = captured["pred"]["verts3d"][0].cpu().numpy()    # (778, 3) root-centered

    return pred_joints, pred_verts


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate HandOCCNet on POV-Surgery")
    parser.add_argument("--mode", choices=["crop"], default="crop",
                        help="crop: GT-derived bboxes")
    parser.add_argument("--split", type=str, default="demo",
                        help="'demo' (s_scalpel_3 only, default), "
                             "'full' (all test sequences via official split), "
                             "or a sequence name (e.g. 'm_diskplacer_1')")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames (0 = all)")
    parser.add_argument("--data-dir", type=str, default="../pov_surgery_data",
                        help="Path to pov_surgery_data/ (default: ../pov_surgery_data)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto based on split)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="HandOCCNet checkpoint path (default: auto-detect)")
    args = parser.parse_args()

    # Resolve paths
    data_dir = Path(args.data_dir).resolve()

    # Determine data root and sequences based on split
    if args.split == "demo":
        data_root = data_dir / "demo_data" / "POV_Surgery_data"
        sequences = ["s_scalpel_3"]
        split_label = "s_scalpel_3 (demo)"
    elif args.split == "full":
        data_root = data_dir / "demo_data" / "POV_Surgery_data"
        test_pkl = data_root / "handoccnet_train" / "2d_repro_ho3d_style_test_cleaned.pkl"
        if not test_pkl.exists():
            print(f"ERROR: Test split pickle not found at {test_pkl}")
            sys.exit(1)
        with open(test_pkl, "rb") as f:
            test_info = pickle.load(f)
        sequences = sorted(set(k.split("/")[0] for k in test_info.keys()))
        split_label = f"full test set ({len(sequences)} sequences)"
    else:
        data_root = data_dir / "demo_data" / "POV_Surgery_data"
        sequences = [args.split]
        split_label = args.split

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
    print(f"Device: {device}")
    print(f"Split: {split_label}")

    # Collect annotation/image files across all sequences
    annotation_files = []
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
                annotation_files.append((seq, pkl_path, img_path))

    if not annotation_files:
        print(f"ERROR: No annotation/image pairs found for split '{args.split}'")
        sys.exit(1)
    print(f"Found {len(annotation_files)} frames across {len(sequences)} sequence(s)")

    if args.max_frames > 0:
        annotation_files = annotation_files[: args.max_frames]
        print(f"Processing first {len(annotation_files)} frames")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Init MANO model for GT
    print("Loading MANO model for GT...")
    gt_mano, tip_ids = build_gt_mano(mano_model_dir, device)

    # Init HandOCCNet model
    print("Loading HandOCCNet model...")
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        # Auto-detect: look in AIM_2_Project/checkpoints/
        project_dir = Path(__file__).resolve().parent.parent
        ckpt_path = project_dir / "checkpoints" / "HandOccNet_model_dump" / "HO3D" / "snapshot_80.pth.tar"
        if not ckpt_path.exists():
            print(f"ERROR: HandOCCNet checkpoint not found at {ckpt_path}")
            print("Download from: https://drive.google.com/drive/folders/1OlyV-qbzOmtQYdzV6dbQX4OtAU5ajBOa")
            print("Or specify --checkpoint /path/to/snapshot.pth.tar")
            sys.exit(1)
    model = load_handoccnet(ckpt_path, device)

    # ── Precompute GT (saved to disk, keyed by seq/frame_id) ─────────
    gt_cache_path = output_dir / f"gt_cache_{args.split}.pkl"
    if gt_cache_path.exists():
        print(f"\nLoading cached GT from {gt_cache_path}...")
        with open(gt_cache_path, "rb") as f:
            gt_cache = pickle.load(f)
        print(f"  Loaded GT cache: {len(gt_cache)} frames.")
        bbox_failures = sum(1 for v in gt_cache.values() if v["bbox"] is None)
    else:
        print("\nPrecomputing GT for all frames...")
        gt_cache = {}
        bbox_failures = 0
        for seq, pkl_path, img_path in annotation_files:
            key = f"{seq}/{pkl_path.stem}"
            gt = load_gt_annotation(pkl_path)
            gt_joints_local, gt_verts_local = mano_forward(
                gt_mano, tip_ids,
                gt["global_orient"], gt["hand_pose"], gt["betas"],
                device,
            )
            gt_joints, gt_verts = transform_to_camera(gt_joints_local, gt_verts_local, gt)
            bbox = derive_bbox_from_gt(gt_joints, K, (1080, 1920), pad_factor=1.5)
            gt_cache[key] = {
                "joints": gt_joints, "verts": gt_verts, "bbox": bbox,
            }
        with open(gt_cache_path, "wb") as f:
            pickle.dump(gt_cache, f)
        print(f"  GT precomputed and saved: {len(gt_cache)} frames -> {gt_cache_path}")

    # ── Evaluation loop ────────────────────────────────────────────────
    print(f"\nMode: {args.mode}")
    all_metrics = []
    inference_failures = 0
    t_start = time.time()

    for i, (seq, pkl_path, img_path) in enumerate(annotation_files):
        frame_id = pkl_path.stem
        key = f"{seq}/{frame_id}"

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

        # Load image
        img_cv2 = cv2.imread(str(img_path))

        # Preprocess for HandOCCNet (crop to 256x256)
        img_tensor, affine_M = handoccnet_preprocess(img_cv2, bbox)

        # Load annotation for model's internal GT pipeline
        anno_data = load_gt_annotation(pkl_path)

        try:
            pred_joints, pred_verts = handoccnet_inference(
                model, img_tensor, anno_data, device)
        except Exception as e:
            inference_failures += 1
            all_metrics.append({"frame_id": key, "detected": False, "reason": f"inference_error: {e}"})
            continue

        # pred_joints and pred_verts are already root-centered from the model
        # GT needs root-centering for comparison
        gt_root_rel = gt_joints - gt_joints[0:1]
        gt_verts_rel = gt_verts - gt_joints[0:1]

        # Root-relative MPJPE
        mpjpe = np.sqrt(((pred_joints - gt_root_rel) ** 2).sum(axis=-1)).mean() * 1000

        # PA-MPJPE
        pa_mpjpe = compute_pa_mpjpe(pred_joints, gt_root_rel)

        # PVE
        pve = np.sqrt(((pred_verts - gt_verts_rel) ** 2).sum(axis=-1)).mean() * 1000

        # PA-PVE
        pred_verts_aligned = procrustes_align(pred_verts, gt_verts_rel)
        pa_pve = np.sqrt(((pred_verts_aligned - gt_verts_rel) ** 2).sum(axis=-1)).mean() * 1000

        # Per-finger
        per_finger = compute_per_finger_mpjpe(pred_joints, gt_root_rel)

        # P2D: HandOCCNet doesn't produce full-image 2D projections easily
        # (its predictions are in MANO-local space, not camera space).
        # We skip P2D for HandOCCNet.
        p2d_error = float("nan")

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
    print(f"RESULTS: HandOCCNet (off-the-shelf, GT-bbox crop-regress)")
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

        print(f"\n{'Metric':<20} {'Mean':>8} {'Median':>8} {'Std':>8}")
        print("-" * 48)
        for name, vals in [
            ("MPJPE (mm)", mpjpe_vals),
            ("PA-MPJPE (mm)", pa_mpjpe_vals),
            ("PVE (mm)", pve_vals),
            ("PA-PVE (mm)", pa_pve_vals),
        ]:
            arr = np.array(vals)
            print(f"{name:<20} {arr.mean():>8.2f} {np.median(arr):>8.2f} {arr.std():>8.2f}")

        # Per-finger
        print(f"\n{'Finger':<12} {'MPJPE (mm)':>12}")
        print("-" * 26)
        for fn in ["thumb", "index", "middle", "ring", "pinky"]:
            vals = [m["per_finger"][fn] for m in detected]
            print(f"{fn:<12} {np.mean(vals):>12.2f}")

        print(f"\n{'HandOCCNet (raw)':<20} MPJPE={np.mean(mpjpe_vals):.2f}  PA-MPJPE={np.mean(pa_mpjpe_vals):.2f}")

    # Save results
    results_path = output_dir / f"handoccnet_{args.mode}_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": f"HandOCCNet (off-the-shelf, {args.mode})",
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
                "p2d_mean": None,
            },
            "per_frame": all_metrics,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
