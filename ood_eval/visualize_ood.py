#!/usr/bin/env python3
"""Visualize OOD evaluation predictions stored in wilor_ood_results.json.

Reads per-sample predictions and GT (if present) from the JSON produced by
run_eval_ood.py and produces per-frame overlay images plus a composite grid.

Usage:
    python visualize_ood.py \
        --results path/to/wilor_ood_results.json \
        --dataset-dir path/to/common_format/<dataset> \
        [--output-dir path/to/figures] [--n-each 2] [--crop-size 512]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGER_COLORS_GT = [
    (0, 180, 0), (0, 210, 50), (0, 230, 100), (0, 200, 150), (0, 170, 190),
]
FINGER_COLORS_PRED = [
    (30, 30, 230), (60, 30, 220), (90, 30, 230), (120, 30, 210), (150, 30, 190),
]


def procrustes_align_2d(pred_2d, gt_2d):
    mu_p = pred_2d.mean(axis=0, keepdims=True)
    mu_g = gt_2d.mean(axis=0, keepdims=True)
    p = pred_2d - mu_p
    g = gt_2d - mu_g

    var_p = np.sum(p ** 2)
    if var_p < 1e-10:
        return pred_2d.copy()

    H = p.T @ g
    U, s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, d])
    R = (Vt.T @ D @ U.T)
    scale = np.trace(R @ H) / var_p
    t = mu_g - scale * (mu_p @ R.T)

    return scale * (pred_2d @ R.T) + t


def draw_skeleton(img, kp2d, skeleton, joint_color, bone_colors,
                  radius=5, thickness=2, alpha=1.0):
    if alpha < 1.0:
        overlay = img.copy()
    else:
        overlay = img

    for idx, (i, j) in enumerate(skeleton):
        color = bone_colors[idx // 4]
        x1, y1 = int(round(kp2d[i, 0])), int(round(kp2d[i, 1]))
        x2, y2 = int(round(kp2d[j, 0])), int(round(kp2d[j, 1]))
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    for i in range(kp2d.shape[0]):
        x, y = int(round(kp2d[i, 0])), int(round(kp2d[i, 1]))
        cv2.circle(overlay, (x, y), radius, joint_color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def crop_to_hand(img, center_pts_list, pad_factor=2.0, crop_size=512):
    """Crop image around hand region defined by stacking all point arrays.

    Returns cropped image, scale applied, and (x1, y1) offset so callers can
    shift any keypoints: kp_new = (kp - [x1, y1]) * scale.
    """
    all_pts = np.vstack([p for p in center_pts_list if p is not None])
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    span = max(x_max - x_min, y_max - y_min) * pad_factor
    span = max(span, 150)

    h, w = img.shape[:2]
    x1 = int(max(0, cx - span / 2))
    y1 = int(max(0, cy - span / 2))
    x2 = int(min(w, cx + span / 2))
    y2 = int(min(h, cy + span / 2))

    cropped = img[y1:y2, x1:x2].copy()
    if cropped.size == 0:
        cropped = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        return cropped, 1.0, (0, 0)

    scale = crop_size / max(cropped.shape[:2])
    cropped = cv2.resize(cropped, (int(cropped.shape[1] * scale),
                                    int(cropped.shape[0] * scale)))
    ch, cw = cropped.shape[:2]
    if ch != crop_size or cw != crop_size:
        padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        padded[:ch, :cw] = cropped
        cropped = padded

    return cropped, scale, (x1, y1)


def _shift(kp, offset, scale):
    return (kp - np.array(offset, dtype=np.float32)) * scale


def _metric_str(frame):
    if frame.get("pa_mpjpe") is not None:
        return f"PA-MPJPE: {frame['pa_mpjpe']:.1f}mm"
    if frame.get("p2d") is not None:
        return f"P2D: {frame['p2d']:.1f}px"
    return "no GT"


def make_three_panel_vis(image_bgr, gt_2d, pred_2d_aligned, frame, crop_size=512):
    """3-panel: GT | Pred (aligned) | Both overlaid."""
    cropped, scale, off = crop_to_hand(
        image_bgr, [gt_2d, pred_2d_aligned], pad_factor=2.0, crop_size=crop_size)
    gt_c = _shift(gt_2d, off, scale)
    pred_c = _shift(pred_2d_aligned, off, scale)

    p1 = cropped.copy()
    draw_skeleton(p1, gt_c, SKELETON, (0, 255, 0), FINGER_COLORS_GT)
    cv2.putText(p1, "GT", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    p2 = cropped.copy()
    draw_skeleton(p2, pred_c, SKELETON, (0, 0, 255), FINGER_COLORS_PRED)
    cv2.putText(p2, "Pred (aligned)", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    p3 = cropped.copy()
    draw_skeleton(p3, gt_c, SKELETON, (0, 255, 0), FINGER_COLORS_GT, radius=4)
    draw_skeleton(p3, pred_c, SKELETON, (0, 0, 255), FINGER_COLORS_PRED, radius=4)

    hand_side = frame.get("hand_side", "right")
    badge = "[LEFT] " if hand_side == "left" else ""
    cv2.putText(p3, badge + _metric_str(frame), (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if frame.get("mpjpe") is not None:
        cv2.putText(p3, f"MPJPE: {frame['mpjpe']:.1f}mm", (8, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(p3, f"#{frame['frame_id']}", (8, crop_size - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    return np.concatenate([p1, p2, p3], axis=1)


def make_single_panel_vis(image_bgr, pred_2d, frame, crop_size=512):
    """Single-panel pred-only visualization (used when GT 2D is absent)."""
    cropped, scale, off = crop_to_hand(
        image_bgr, [pred_2d], pad_factor=2.0, crop_size=crop_size)
    pred_c = _shift(pred_2d, off, scale)

    panel = cropped.copy()
    draw_skeleton(panel, pred_c, SKELETON, (0, 0, 255), FINGER_COLORS_PRED)

    hand_side = frame.get("hand_side", "right")
    badge = "[LEFT] " if hand_side == "left" else ""
    cv2.putText(panel, badge + "Pred", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, _metric_str(frame), (8, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"#{frame['frame_id']}", (8, crop_size - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return panel


def _score(frame):
    if frame.get("pa_mpjpe") is not None:
        return frame["pa_mpjpe"]
    if frame.get("p2d") is not None:
        return frame["p2d"]
    return None


def select_frames_scored(detected, n_each=2):
    """Select n best/median/worst/spread using _score as criterion."""
    n = len(detected)
    sorted_indices = sorted(range(n), key=lambda i: _score(detected[i]))

    selected = {}
    for i in sorted_indices[:n_each]:
        selected[i] = "best"
    for i in sorted_indices[-n_each:]:
        selected[i] = "worst"

    mid = n // 2
    for offset in range(n_each):
        idx = mid + offset - n_each // 2
        idx = max(0, min(n - 1, idx))
        if idx not in selected:
            selected[idx] = "median"
        else:
            for delta in range(1, n):
                for candidate in [mid + offset + delta, mid + offset - delta]:
                    if 0 <= candidate < n and candidate not in selected:
                        selected[candidate] = "median"
                        break
                if sum(1 for v in selected.values() if v == "median") >= n_each:
                    break

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

    result = sorted(selected.items(), key=lambda x: _score(detected[x[0]]))
    return result


def select_frames_spread(detected, n_total=8):
    """Evenly spread selection when no GT score available."""
    n = len(detected)
    n_total = min(n_total, n)
    if n_total == 0:
        return []
    step = max(1, n // n_total)
    idxs = []
    for k in range(n_total):
        idx = min(n - 1, k * step)
        if idx not in idxs:
            idxs.append(idx)
    return [(i, "spread") for i in idxs]


def make_composite_grid(panels, categories, frames, output_path, dataset_name):
    n = len(panels)
    if n == 0:
        return
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    cat_colors = {"best": "#2ecc71", "median": "#f39c12",
                  "worst": "#e74c3c", "spread": "#3498db"}

    for i, (panel, frame, cat) in enumerate(zip(panels, frames, categories)):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])

        hand_side = frame.get("hand_side", "right")
        badge = "[L] " if hand_side == "left" else ""
        score = _score(frame)
        if score is not None:
            label = "PA-MPJPE" if frame.get("pa_mpjpe") is not None else "P2D"
            title = f"{badge}#{frame['frame_id']}  {label}={score:.1f}"
        else:
            title = f"{badge}#{frame['frame_id']}"
        ax.set_title(title, fontsize=10, color=cat_colors.get(cat, "black"),
                     fontweight="bold")

        ax.text(0.02, 0.02, cat.upper(), transform=ax.transAxes,
                fontsize=8, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor=cat_colors.get(cat, "gray"), alpha=0.85),
                verticalalignment="bottom")

    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    plt.suptitle(f"GT (green) vs Pred (red, Procrustes-aligned)\n{dataset_name}",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved composite grid: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize OOD evaluation predictions (from JSON).")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to wilor_ood_results.json")
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Path to common_format/<dataset> (samples.pkl parent)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir (default: <results_dir>/figures/sample_overlays/)")
    parser.add_argument("--n-each", type=int, default=2,
                        help="Frames per category for scored selection (default: 2)")
    parser.add_argument("--crop-size", type=int, default=512,
                        help="Crop size in pixels (default: 512)")
    args = parser.parse_args()

    results_path = Path(args.results).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_path.parent / "figures" / "sample_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        data = json.load(f)
    per_sample = data.get("per_sample", [])
    detected = [m for m in per_sample
                if m.get("detected", False) and m.get("pred_joints_2d") is not None]
    if not detected:
        print("ERROR: No samples with predicted 2D joints.")
        return

    has_gt_2d = any(m.get("gt_joints_2d") is not None for m in detected)
    has_gt_3d = any(m.get("gt_joints_3d") is not None for m in detected)
    has_score = any(_score(m) is not None for m in detected)

    dataset_name = data.get("dataset", dataset_dir.name)
    print(f"Dataset: {dataset_name}  (has_gt_2d={has_gt_2d}, has_gt_3d={has_gt_3d})")
    print(f"Detected samples: {len(detected)}")

    if has_score:
        selection = select_frames_scored(detected, n_each=args.n_each)
    else:
        selection = select_frames_spread(detected, n_total=args.n_each * 4)

    total = len(selection)
    cats_count = {"best": 0, "median": 0, "worst": 0, "spread": 0}
    for _, c in selection:
        cats_count[c] = cats_count.get(c, 0) + 1
    print(f"Selected {total} frames: "
          f"{cats_count.get('best',0)} best, {cats_count.get('median',0)} median, "
          f"{cats_count.get('worst',0)} worst, {cats_count.get('spread',0)} spread")

    grid_panels = []
    grid_frames = []
    grid_categories = []

    for count, (idx, cat) in enumerate(selection):
        frame = detected[idx]
        frame_id = frame["frame_id"]
        rel_img = frame["image_path"]
        img_path = (dataset_dir / rel_img).resolve()

        if not img_path.exists():
            print(f"  WARNING: image not found: {img_path}")
            continue

        image_rgb = np.array(Image.open(img_path).convert("RGB"))
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        pred_2d = np.asarray(frame["pred_joints_2d"], dtype=np.float32)
        gt_2d = frame.get("gt_joints_2d")
        if gt_2d is not None:
            gt_2d = np.asarray(gt_2d, dtype=np.float32)

        safe_id = str(frame_id).replace("/", "_").replace(" ", "_")

        if gt_2d is not None:
            pred_2d_aligned = procrustes_align_2d(pred_2d, gt_2d)
            combined = make_three_panel_vis(
                img_bgr, gt_2d, pred_2d_aligned, frame, crop_size=args.crop_size)
            out_path = output_dir / f"{safe_id}_{cat}_overlay.jpg"
            cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 95])

            cropped, scale, off = crop_to_hand(
                img_bgr, [gt_2d, pred_2d_aligned],
                pad_factor=2.0, crop_size=args.crop_size)
            gt_c = _shift(gt_2d, off, scale)
            pred_c = _shift(pred_2d_aligned, off, scale)
            grid_panel = cropped.copy()
            draw_skeleton(grid_panel, gt_c, SKELETON, (0, 255, 0),
                          FINGER_COLORS_GT, radius=4)
            draw_skeleton(grid_panel, pred_c, SKELETON, (0, 0, 255),
                          FINGER_COLORS_PRED, radius=4)
        else:
            panel = make_single_panel_vis(
                img_bgr, pred_2d, frame, crop_size=args.crop_size)
            out_path = output_dir / f"{safe_id}_{cat}_overlay.jpg"
            cv2.imwrite(str(out_path), panel, [cv2.IMWRITE_JPEG_QUALITY, 95])

            cropped, scale, off = crop_to_hand(
                img_bgr, [pred_2d], pad_factor=2.0, crop_size=args.crop_size)
            pred_c = _shift(pred_2d, off, scale)
            grid_panel = cropped.copy()
            draw_skeleton(grid_panel, pred_c, SKELETON, (0, 0, 255),
                          FINGER_COLORS_PRED, radius=4)

        grid_panels.append(grid_panel)
        grid_frames.append(frame)
        grid_categories.append(cat)

        score = _score(frame)
        hand_tag = " [LEFT]" if frame.get("hand_side") == "left" else ""
        score_str = f"{score:.1f}" if score is not None else "no-GT"
        print(f"  [{count+1}/{total}] {frame_id} ({cat}){hand_tag}: "
              f"score={score_str}  -> {out_path.name}")

    if grid_panels:
        grid_path = output_dir.parent / "prediction_grid.png"
        make_composite_grid(grid_panels, grid_categories, grid_frames,
                            grid_path, dataset_name)

    print(f"\nDone! {len(grid_panels)} overlays saved to {output_dir}/")


if __name__ == "__main__":
    main()
