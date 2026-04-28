#!/usr/bin/env python3
"""Overlay GT vs predicted hand keypoints/skeleton on POV-Surgery frames.

Generates two types of output:
  1. Individual per-frame overlays (cropped to hand region)
  2. A composite grid figure for papers/presentations

For each frame, shows:
  - Cropped image with GT skeleton (green)
  - Same crop with WiLoR predicted skeleton (red), Procrustes-aligned to GT in 2D
  - Both overlaid so you can directly compare finger articulation

Frame selection: picks 2 best + 2 median + 2 worst + 2 spread (8 total by default).

Usage:
    python visualize_predictions.py --results results/wilor_crop_results.json --data-dir ../pov_surgery_data
    python visualize_predictions.py --results results/wilor_crop_results.json --data-dir ../pov_surgery_data --n-each 3
"""

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# POV-Surgery camera intrinsics
K = np.array([
    [1198.4395, 0.0, 960.0],
    [0.0, 1198.4395, 175.2],
    [0.0, 0.0, 1.0],
])

COORD_CHANGE = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
]

# Distinct per-finger colors (BGR for cv2)
FINGER_COLORS_GT = [
    (0, 180, 0), (0, 210, 50), (0, 230, 100), (0, 200, 150), (0, 170, 190),
]
FINGER_COLORS_PRED = [
    (30, 30, 230), (60, 30, 220), (90, 30, 230), (120, 30, 210), (150, 30, 190),
]


# ── MANO / projection helpers ─────────────────────────────────────────

def build_gt_mano(mano_model_dir, device):
    import smplx
    from smplx.vertex_ids import vertex_ids as smplx_vertex_ids
    mano = smplx.create(
        str(mano_model_dir), model_type="mano",
        is_rhand=True, use_pca=False, flat_hand_mean=True,
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
    joints_21 = np.concatenate([joints_16, tips])[MANO_TO_OPENPOSE]
    return joints_21, vertices


def transform_to_camera(joints, verts, anno):
    cam_rot, cam_transl = anno["cam_rot"], anno["cam_transl"]
    g2w_R, g2w_T, transl = anno["grab2world_R"], anno["grab2world_T"], anno["transl"]
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = cam_transl
    camera_pose[:3, :3] = cam_rot
    cam_inv = np.linalg.inv(camera_pose)
    all_g = g2w_R @ cam_inv[:3, :3].T
    all_t = g2w_T @ cam_inv[:3, :3].T + cam_inv[:3, 3]
    all_t_no = transl @ all_g + all_t
    j_cam = joints @ all_g + all_t_no
    v_cam = verts @ all_g + all_t_no
    return j_cam @ COORD_CHANGE.T, v_cam @ COORD_CHANGE.T


def project_to_2d(joints_3d, K):
    pts = (K @ joints_3d.T).T
    return pts[:, :2] / pts[:, 2:3]


def load_gt_annotation(pkl_path):
    with open(pkl_path, "rb") as f:
        anno = pickle.load(f)
    m = anno["mano"]
    return {
        "global_orient": np.array(m["global_orient"], dtype=np.float32),
        "hand_pose": np.array(m["hand_pose"], dtype=np.float32),
        "betas": np.array(m["betas"], dtype=np.float32),
        "transl": np.array(m["transl"], dtype=np.float32),
        "cam_rot": np.array(anno.get("cam_rot", np.eye(3)), dtype=np.float64),
        "cam_transl": np.array(anno.get("cam_transl", np.zeros(3)), dtype=np.float64),
        "grab2world_R": np.array(anno.get("grab2world_R", np.eye(3)), dtype=np.float64),
        "grab2world_T": np.array(anno.get("grab2world_T", np.zeros((1, 3))), dtype=np.float64),
    }


# ── 2D Procrustes alignment ───────────────────────────────────────────

def procrustes_align_2d(pred_2d, gt_2d):
    """Align pred 2D keypoints to GT 2D via similarity transform (rotation + scale + translation).

    This lets us visually compare *shape* (finger articulation) independently of
    WiLoR's broken global translation.
    """
    mu_p = pred_2d.mean(axis=0, keepdims=True)
    mu_g = gt_2d.mean(axis=0, keepdims=True)
    p = pred_2d - mu_p
    g = gt_2d - mu_g

    var_p = np.sum(p ** 2)
    if var_p < 1e-10:
        return pred_2d.copy()

    H = p.T @ g  # (2, 2)
    U, s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, d])
    R = (Vt.T @ D @ U.T)
    scale = np.trace(R @ H) / var_p
    t = mu_g - scale * (mu_p @ R.T)

    return scale * (pred_2d @ R.T) + t


# ── Drawing helpers ────────────────────────────────────────────────────

def draw_skeleton(img, kp2d, skeleton, joint_color, bone_colors,
                  radius=5, thickness=2, alpha=1.0):
    """Draw keypoints and skeleton. If alpha < 1, blends onto img."""
    if alpha < 1.0:
        overlay = img.copy()
    else:
        overlay = img
    h, w = img.shape[:2]

    for idx, (i, j) in enumerate(skeleton):
        color = bone_colors[idx // 4]
        x1, y1 = int(round(kp2d[i, 0])), int(round(kp2d[i, 1]))
        x2, y2 = int(round(kp2d[j, 0])), int(round(kp2d[j, 1]))
        # Allow drawing slightly outside bounds for partial visibility
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    for i in range(kp2d.shape[0]):
        x, y = int(round(kp2d[i, 0])), int(round(kp2d[i, 1]))
        cv2.circle(overlay, (x, y), radius, joint_color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def crop_to_hand(img, gt_2d, pred_2d_aligned, pad_factor=2.0, crop_size=512):
    """Crop image around GT hand region. Returns cropped image and shifted keypoints."""
    all_pts = np.vstack([gt_2d, pred_2d_aligned])
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    span = max(x_max - x_min, y_max - y_min) * pad_factor
    span = max(span, 150)  # minimum crop size

    h, w = img.shape[:2]
    x1 = int(max(0, cx - span / 2))
    y1 = int(max(0, cy - span / 2))
    x2 = int(min(w, cx + span / 2))
    y2 = int(min(h, cy + span / 2))

    cropped = img[y1:y2, x1:x2].copy()
    gt_shifted = gt_2d - np.array([x1, y1])
    pred_shifted = pred_2d_aligned - np.array([x1, y1])

    # Resize to uniform size
    scale = crop_size / max(cropped.shape[:2])
    cropped = cv2.resize(cropped, (int(cropped.shape[1] * scale), int(cropped.shape[0] * scale)))
    # Pad to square
    ch, cw = cropped.shape[:2]
    if ch != crop_size or cw != crop_size:
        padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        padded[:ch, :cw] = cropped
        cropped = padded

    gt_shifted *= scale
    pred_shifted *= scale

    return cropped, gt_shifted, pred_shifted


def make_single_frame_vis(image_bgr, gt_2d, pred_2d_raw, pred_2d_aligned,
                          pa_mpjpe, mpjpe, frame_id, crop_size=512):
    """Create a 3-panel visualization for one frame (all cropped to hand region).

    Panels: GT only | Pred (aligned) only | Both overlaid
    """
    # Crop around hand
    cropped, gt_c, pred_c = crop_to_hand(
        image_bgr, gt_2d, pred_2d_aligned, pad_factor=2.0, crop_size=crop_size)

    # Panel 1: GT
    p1 = cropped.copy()
    draw_skeleton(p1, gt_c, SKELETON, (0, 255, 0), FINGER_COLORS_GT, radius=5, thickness=2)
    cv2.putText(p1, "GT", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Panel 2: Pred (aligned to GT position)
    p2 = cropped.copy()
    draw_skeleton(p2, pred_c, SKELETON, (0, 0, 255), FINGER_COLORS_PRED, radius=5, thickness=2)
    cv2.putText(p2, "Pred (aligned)", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Panel 3: Both overlaid
    p3 = cropped.copy()
    draw_skeleton(p3, gt_c, SKELETON, (0, 255, 0), FINGER_COLORS_GT, radius=4, thickness=2)
    draw_skeleton(p3, pred_c, SKELETON, (0, 0, 255), FINGER_COLORS_PRED, radius=4, thickness=2)
    cv2.putText(p3, f"PA-MPJPE: {pa_mpjpe:.1f}mm", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(p3, f"MPJPE: {mpjpe:.1f}mm", (8, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(p3, f"#{frame_id}", (8, crop_size - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    combined = np.concatenate([p1, p2, p3], axis=1)
    return combined


def select_frames(detected, n_each=2):
    """Select frames: n best, n median, n worst, n spread across sequence.

    Returns list of (index, category_label) tuples.
    """
    n = len(detected)
    sorted_indices = sorted(range(n), key=lambda i: detected[i]["pa_mpjpe"])

    selected = {}

    # Best (lowest PA-MPJPE)
    for i in sorted_indices[:n_each]:
        selected[i] = "best"

    # Worst (highest PA-MPJPE)
    for i in sorted_indices[-n_each:]:
        selected[i] = "worst"

    # Median
    mid = n // 2
    for offset in range(n_each):
        idx = mid + offset - n_each // 2
        idx = max(0, min(n - 1, idx))
        if idx not in selected:
            selected[idx] = "median"
        else:
            # Find nearest unselected
            for delta in range(1, n):
                for candidate in [mid + offset + delta, mid + offset - delta]:
                    if 0 <= candidate < n and candidate not in selected:
                        selected[candidate] = "median"
                        break
                if sum(1 for v in selected.values() if v == "median") >= n_each:
                    break

    # Spread (evenly spaced through sequence, fill remaining)
    n_spread = n_each
    spread_step = max(1, n // (n_spread + 1))
    for k in range(1, n_spread + 1):
        idx = k * spread_step
        idx = min(idx, n - 1)
        if idx not in selected:
            selected[idx] = "spread"
        else:
            for delta in range(1, n):
                for candidate in [idx + delta, idx - delta]:
                    if 0 <= candidate < n and candidate not in selected:
                        selected[candidate] = "spread"
                        break
                if sum(1 for v in selected.values() if v == "spread") >= n_spread:
                    break

    # Sort by PA-MPJPE for display (best to worst)
    result = sorted(selected.items(), key=lambda x: detected[x[0]]["pa_mpjpe"])
    return result


def make_composite_grid(panels, categories, detected, selected_indices, output_path):
    """Create a matplotlib composite grid figure for papers."""
    n = len(panels)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    cat_colors = {"best": "#2ecc71", "median": "#f39c12", "worst": "#e74c3c", "spread": "#3498db"}

    for i, (panel, (idx, cat)) in enumerate(zip(panels, zip(selected_indices, categories))):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        # panel is BGR, convert to RGB for matplotlib
        ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])

        frame = detected[idx]
        title = f"#{frame['frame_id']}  PA-MPJPE={frame['pa_mpjpe']:.1f}mm"
        ax.set_title(title, fontsize=10, color=cat_colors.get(cat, "black"), fontweight="bold")

        # Category badge
        ax.text(0.02, 0.02, cat.upper(), transform=ax.transAxes,
                fontsize=8, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=cat_colors.get(cat, "gray"), alpha=0.85),
                verticalalignment="bottom")

    # Hide unused axes
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    plt.suptitle("GT (green) vs Pred (red, Procrustes-aligned)\nPOV-Surgery s_scalpel_3",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved composite grid: {output_path.name}")


def derive_bbox_from_gt_2d(gt_2d, img_shape, pad_factor=1.5):
    """Derive bounding box from 2D GT keypoints."""
    x_min, y_min = gt_2d.min(axis=0)
    x_max, y_max = gt_2d.max(axis=0)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    s = max(x_max - x_min, y_max - y_min) * pad_factor
    return np.array([
        max(0, cx - s / 2), max(0, cy - s / 2),
        min(img_shape[1], cx + s / 2), min(img_shape[0], cy + s / 2)])


def load_wilor(device):
    """Load WiLoR pipeline."""
    _original_load = torch.load
    def _patched_load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _original_load(*a, **kw)
    torch.load = _patched_load
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    torch.load = _original_load
    return pipe


def predict_wilor_2d(pipe, image_rgb, bbox):
    """Run WiLoR prediction, return raw 2D keypoints."""
    outputs = pipe.predict_with_bboxes(image_rgb, np.array([bbox]), [1.0], rescale_factor=2.5)
    if not outputs:
        return None
    pred_kp2d = outputs[0]["wilor_preds"].get("pred_keypoints_2d")
    if pred_kp2d is not None:
        return pred_kp2d[0]
    pred_3d = outputs[0]["wilor_preds"]["pred_keypoints_3d"][0]
    return project_to_2d(pred_3d, K)


def load_hamer_model(device):
    """Load HaMeR model and config."""
    import os
    project_root = Path(__file__).resolve().parent.parent

    import hamer.configs
    hamer.configs.CACHE_DIR_HAMER = str(project_root / "_DATA")

    from hamer.models import load_hamer
    hamer_ckpt = project_root / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
    model, model_cfg = load_hamer(str(hamer_ckpt))
    model = model.to(device)
    model.eval()
    return model, model_cfg


def predict_hamer_2d(hamer_model, model_cfg, img_bgr, bbox, device):
    """Run HaMeR prediction, return raw 2D keypoints in full-image pixel coords."""
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.utils import recursive_to

    boxes = np.array([bbox])
    right = np.array([1.0])
    dataset = ViTDetDataset(model_cfg, img_bgr, boxes, right, rescale_factor=2.5)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
    batch = recursive_to(batch, device)

    with torch.no_grad():
        out = hamer_model(batch)

    pred_joints = out["pred_keypoints_3d"][0].cpu().numpy()
    pred_cam = out["pred_cam"][0].cpu().numpy()
    box_center = batch["box_center"][0].cpu().numpy()
    img_w, img_h = batch["img_size"][0].cpu().numpy()
    img_max = max(img_w, img_h)
    scaled_fl = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_max

    cam_t = np.array([
        pred_cam[1], pred_cam[2],
        2 * scaled_fl / (model_cfg.MODEL.IMAGE_SIZE * pred_cam[0] + 1e-9),
    ])
    cam_t[:2] += (2 * box_center - np.array([img_w, img_h])) / (
        model_cfg.MODEL.IMAGE_SIZE * pred_cam[0] + 1e-9)

    joints_cam = pred_joints + cam_t[None, :]
    pred_2d = np.zeros((21, 2))
    pred_2d[:, 0] = scaled_fl * joints_cam[:, 0] / joints_cam[:, 2] + img_w / 2
    pred_2d[:, 1] = scaled_fl * joints_cam[:, 1] / joints_cam[:, 2] + img_h / 2
    return pred_2d


def main():
    parser = argparse.ArgumentParser(description="Visualize GT vs predicted hand poses")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON")
    parser.add_argument("--data-dir", type=str, default="../pov_surgery_data",
                        help="Path to pov_surgery_data/")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir (default: results/figures/sample_overlays/)")
    parser.add_argument("--n-each", type=int, default=2,
                        help="Number of frames per category: best/median/worst/spread (default: 2)")
    parser.add_argument("--crop-size", type=int, default=512,
                        help="Size of cropped hand region in pixels (default: 512)")
    parser.add_argument("--model", choices=["wilor", "hamer"], default="wilor",
                        help="Which model to run for prediction overlays (default: wilor)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    results_path = Path(args.results)
    data_dir = Path(args.data_dir).resolve()
    demo_dir = data_dir / "demo_data" / "POV_Surgery_data"
    annotation_dir = demo_dir / "annotation" / "s_scalpel_3"
    image_dir = demo_dir / "color" / "s_scalpel_3"
    mano_model_dir = data_dir / "data" / "bodymodel"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_path.parent / "figures" / "sample_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    with open(results_path) as f:
        data = json.load(f)
    detected = [m for m in data["per_frame"] if m.get("detected", False)]
    if not detected:
        print("ERROR: No detected frames.")
        return

    # Select frames
    selection = select_frames(detected, n_each=args.n_each)
    selected_indices = [idx for idx, _ in selection]
    categories = [cat for _, cat in selection]
    total = len(selection)
    print(f"Selected {total} frames: "
          f"{sum(1 for c in categories if c == 'best')} best, "
          f"{sum(1 for c in categories if c == 'median')} median, "
          f"{sum(1 for c in categories if c == 'worst')} worst, "
          f"{sum(1 for c in categories if c == 'spread')} spread")

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif args.model == "hamer":
        # HaMeR needs CPU (float64 not supported on MPS)
        device = torch.device("cpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model_name = args.model.upper()

    # Load models
    print("Loading MANO model...")
    gt_mano, tip_ids = build_gt_mano(mano_model_dir, device)

    wilor_pipe = None
    hamer_model = None
    hamer_cfg = None

    if args.model == "wilor":
        print("Loading WiLoR pipeline...")
        wilor_pipe = load_wilor(device)
    else:
        print("Loading HaMeR model...")
        hamer_model, hamer_cfg = load_hamer_model(device)
    print("Models loaded.\n")

    # Process each selected frame
    grid_panels = []
    valid_indices = []
    valid_categories = []

    for count, (idx, cat) in enumerate(selection):
        frame = detected[idx]
        frame_id = frame["frame_id"]
        img_path = image_dir / f"{frame_id}.jpg"
        pkl_path = annotation_dir / f"{frame_id}.pkl"

        if not img_path.exists() or not pkl_path.exists():
            print(f"  Skipping {frame_id}: missing files")
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        gt = load_gt_annotation(pkl_path)

        # GT: MANO FK -> camera -> 2D
        gt_j_local, gt_v_local = mano_forward(
            gt_mano, tip_ids,
            gt["global_orient"], gt["hand_pose"], gt["betas"], device)
        gt_j_cam, _ = transform_to_camera(gt_j_local, gt_v_local, gt)
        gt_2d = project_to_2d(gt_j_cam, K)

        # Derive bbox from GT 2D projections
        bbox_pts = (K @ gt_j_cam.T).T
        z = bbox_pts[:, 2]
        valid_mask = z > 0.01
        if valid_mask.sum() < 5:
            print(f"  Skipping {frame_id}: projection failed")
            continue
        pts_2d_bbox = bbox_pts[valid_mask, :2] / bbox_pts[valid_mask, 2:3]
        bbox = derive_bbox_from_gt_2d(pts_2d_bbox, image.shape[:2], pad_factor=1.5)

        # Run model prediction for 2D keypoints
        if args.model == "wilor":
            pred_2d_raw = predict_wilor_2d(wilor_pipe, image, bbox)
        else:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            try:
                pred_2d_raw = predict_hamer_2d(hamer_model, hamer_cfg, img_bgr, bbox, device)
            except Exception as e:
                print(f"  Skipping {frame_id}: {e}")
                continue

        if pred_2d_raw is None:
            print(f"  Skipping {frame_id}: no {model_name} output")
            continue

        # Procrustes-align pred 2D onto GT 2D (removes position/scale offset)
        pred_2d_aligned = procrustes_align_2d(pred_2d_raw, gt_2d)

        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Individual frame overlay (3 panels)
        combined = make_single_frame_vis(
            img_bgr, gt_2d, pred_2d_raw, pred_2d_aligned,
            frame["pa_mpjpe"], frame["mpjpe"], frame_id,
            crop_size=args.crop_size)
        out_path = output_dir / f"{frame_id}_{cat}_overlay.jpg"
        cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # For composite grid: just the "both overlaid" crop
        cropped, gt_c, pred_c = crop_to_hand(
            img_bgr, gt_2d, pred_2d_aligned, pad_factor=2.0, crop_size=args.crop_size)
        grid_panel = cropped.copy()
        draw_skeleton(grid_panel, gt_c, SKELETON, (0, 255, 0), FINGER_COLORS_GT, radius=4, thickness=2)
        draw_skeleton(grid_panel, pred_c, SKELETON, (0, 0, 255), FINGER_COLORS_PRED, radius=4, thickness=2)
        grid_panels.append(grid_panel)
        valid_indices.append(idx)
        valid_categories.append(cat)

        print(f"  [{count+1}/{total}] {frame_id} ({cat}): "
              f"PA-MPJPE={frame['pa_mpjpe']:.1f}mm  -> {out_path.name}")

    # Composite grid
    if grid_panels:
        grid_path = output_dir.parent / "prediction_grid.png"
        make_composite_grid(grid_panels, valid_categories, detected,
                            valid_indices, grid_path)

    print(f"\nDone! {len(grid_panels)} overlays saved to {output_dir}/")


if __name__ == "__main__":
    main()
