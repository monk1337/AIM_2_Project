#!/usr/bin/env python3
"""Side-by-side comparison of raw vs finetuned WiLoR on the same frames.

Loads two wilor_ood_results.json files (raw + finetuned), matches frames by
frame_id, and emits a composite grid where each row is one frame showing
raw prediction on the left, finetuned prediction on the right.

Frame selection:
  - When GT is available: rank frames by (raw.pa_mpjpe - ft.pa_mpjpe). Pick the
    top --n-wins (finetuned wins the most) and the bottom --n-losses (raw
    wins the most / finetuned worst relative to raw).
  - When no GT: just pick evenly-spread frames shared between the two runs.

Usage:
    python compare_raw_vs_ft.py \
        --raw results/aria_raw/wilor_ood_results.json \
        --ft  results/aria_finetuned/wilor_ood_results.json \
        --dataset-dir common_format/aria \
        --output-dir results/compare_aria \
        [--n-wins 4 --n-losses 4] [--crop-size 512]
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

from visualize_ood import (
    SKELETON, FINGER_COLORS_GT, FINGER_COLORS_PRED,
    procrustes_align_2d, draw_skeleton, crop_to_hand, _shift,
)


def index_by_frame_id(per_sample):
    return {s["frame_id"]: s for s in per_sample}


def _metric(f, key):
    return f.get(key)


def select_paired_frames(raw_by_id, ft_by_id, n_wins, n_losses, rank_by):
    """Select shared-frame indices by delta = raw[rank_by] - ft[rank_by].

    rank_by: "p2d" (pixel error, matches visualization) or "pa_mpjpe" (3D shape).

    Returns list of (frame_id, category, delta) where category is
    "ft_wins" or "raw_wins". Sorted so ft_wins come first (ranked by biggest
    improvement), then raw_wins (ranked by biggest regression).
    """
    paired = []
    for fid, r in raw_by_id.items():
        f = ft_by_id.get(fid)
        if f is None:
            continue
        if not r.get("detected") or not f.get("detected"):
            continue
        pr, pf = _metric(r, rank_by), _metric(f, rank_by)
        if pr is None or pf is None:
            continue
        if r.get("pred_joints_2d") is None or f.get("pred_joints_2d") is None:
            continue
        paired.append((fid, pr - pf))  # positive => ft wins

    if not paired:
        return []

    paired.sort(key=lambda x: x[1], reverse=True)  # ft wins first

    wins = paired[:n_wins]
    losses = paired[-n_losses:] if n_losses > 0 else []
    losses.reverse()  # worst regression first

    out = [(fid, "ft_wins", d) for fid, d in wins]
    out += [(fid, "raw_wins", d) for fid, d in losses]
    return out


def select_spread_frames(raw_by_id, ft_by_id, n_total):
    """Evenly spread shared frames when no PA-MPJPE is available."""
    shared = [fid for fid in raw_by_id if fid in ft_by_id
              and raw_by_id[fid].get("pred_joints_2d") is not None
              and ft_by_id[fid].get("pred_joints_2d") is not None]
    if not shared:
        return []
    n_total = min(n_total, len(shared))
    step = max(1, len(shared) // n_total)
    idxs = []
    for k in range(n_total):
        idx = min(len(shared) - 1, k * step)
        if idx not in idxs:
            idxs.append(idx)
    return [(shared[i], "spread", 0.0) for i in idxs]


def render_pair(image_bgr, raw_sample, ft_sample, crop_size=512, has_gt=True,
                align_2d=False):
    """Render a 2-panel row: [raw pred] [ft pred] on the same crop.

    If align_2d is True, predictions are Procrustes-aligned to GT in 2D
    (shows shape only). Otherwise predictions are drawn in their raw image
    pixel coordinates (shows what the model actually projects to).
    """
    raw_pred = np.asarray(raw_sample["pred_joints_2d"], dtype=np.float32)
    ft_pred = np.asarray(ft_sample["pred_joints_2d"], dtype=np.float32)
    gt = raw_sample.get("gt_joints_2d")
    if gt is not None:
        gt = np.asarray(gt, dtype=np.float32)

    if has_gt and gt is not None and align_2d:
        raw_draw = procrustes_align_2d(raw_pred, gt)
        ft_draw = procrustes_align_2d(ft_pred, gt)
        pts_for_crop = [gt, raw_draw, ft_draw]
    elif has_gt and gt is not None:
        raw_draw = raw_pred
        ft_draw = ft_pred
        pts_for_crop = [gt, raw_draw, ft_draw]
    else:
        raw_draw = raw_pred
        ft_draw = ft_pred
        pts_for_crop = [raw_draw, ft_draw]

    cropped, scale, off = crop_to_hand(image_bgr, pts_for_crop,
                                       pad_factor=2.0, crop_size=crop_size)

    panels = []
    for label, pred, sample in (("Raw", raw_draw, raw_sample),
                                 ("Finetuned", ft_draw, ft_sample)):
        panel = cropped.copy()
        pred_c = _shift(pred, off, scale)
        if has_gt and gt is not None:
            gt_c = _shift(gt, off, scale)
            draw_skeleton(panel, gt_c, SKELETON, (0, 255, 0), FINGER_COLORS_GT,
                          radius=4, thickness=2)
        draw_skeleton(panel, pred_c, SKELETON, (0, 0, 255), FINGER_COLORS_PRED,
                      radius=4, thickness=2)

        cv2.putText(panel, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        if has_gt:
            parts = []
            if sample.get("p2d") is not None:
                parts.append(f"P2D={sample['p2d']:.1f}px")
            if sample.get("pa_mpjpe") is not None:
                parts.append(f"PA-MPJPE={sample['pa_mpjpe']:.1f}mm")
            if parts:
                cv2.putText(panel, " | ".join(parts), (8, crop_size - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)

    return panels


def make_composite_grid(rows, categories, titles, output_path, dataset_name,
                        has_gt, n_wins, n_losses, rank_by, align_2d):
    """rows: list of (panel_raw, panel_ft). One row per frame."""
    n = len(rows)
    if n == 0:
        print("No rows to render.")
        return

    fig, axes = plt.subplots(n, 2, figsize=(9, 4.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    cat_colors = {
        "ft_wins": "#2ecc71",   # green: finetuned wins
        "raw_wins": "#e74c3c",  # red: raw wins / ft regression
        "spread": "#3498db",
    }

    for i, ((p_raw, p_ft), cat, title) in enumerate(zip(rows, categories, titles)):
        for j, panel in enumerate([p_raw, p_ft]):
            ax = axes[i, j]
            ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.text(0.02, 0.02, cat.upper().replace("_", " "),
                        transform=ax.transAxes, fontsize=9, color="white",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor=cat_colors.get(cat, "gray"),
                                  alpha=0.85),
                        verticalalignment="bottom")
        axes[i, 0].set_title(title, fontsize=10, loc="left", fontweight="bold",
                             color=cat_colors.get(cat, "black"))

    axes[0, 0].set_title(f"Raw WiLoR\n{axes[0, 0].get_title()}",
                         fontsize=11, loc="left", fontweight="bold")
    axes[0, 1].set_title("Finetuned WiLoR", fontsize=11, fontweight="bold")

    header = f"Raw vs Finetuned WiLoR — {dataset_name}"
    if has_gt:
        header += f"  |  {n_wins} ft-wins + {n_losses} raw-wins  (ranked by {rank_by})"
        pred_note = "Procrustes-aligned in 2D" if align_2d else "image-pixel coords"
        header += f"\nGT (green) + prediction (red, {pred_note})"
    else:
        header += " (no GT — qualitative only; red = prediction)"
    plt.suptitle(header, fontsize=13, y=1.0)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--ft", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-wins", type=int, default=4)
    parser.add_argument("--n-losses", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--n-spread", type=int, default=8,
                        help="Frames to show for no-GT datasets")
    parser.add_argument("--rank-by", choices=["p2d", "pa_mpjpe"], default="p2d",
                        help="Metric used to rank ft-wins vs raw-wins frames "
                             "(default p2d — matches the 2D visualization)")
    parser.add_argument("--align-2d", action="store_true",
                        help="Procrustes-align predictions to GT in 2D before "
                             "drawing (shows shape only, hides position error)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.raw) as f:
        raw_data = json.load(f)
    with open(args.ft) as f:
        ft_data = json.load(f)

    dataset_name = raw_data.get("dataset", dataset_dir.name)
    print(f"Dataset: {dataset_name}")
    print(f"  raw   samples: {len(raw_data['per_sample'])}")
    print(f"  ft    samples: {len(ft_data['per_sample'])}")

    raw_by_id = index_by_frame_id(raw_data["per_sample"])
    ft_by_id = index_by_frame_id(ft_data["per_sample"])

    has_gt = any(s.get("gt_joints_2d") is not None
                 and s.get(args.rank_by) is not None
                 for s in raw_data["per_sample"])
    print(f"  has GT + {args.rank_by}: {has_gt}")

    if has_gt:
        selection = select_paired_frames(raw_by_id, ft_by_id,
                                         args.n_wins, args.n_losses,
                                         args.rank_by)
    else:
        selection = select_spread_frames(raw_by_id, ft_by_id, args.n_spread)

    if not selection:
        print("ERROR: no valid paired frames found.")
        return

    print(f"Selected {len(selection)} frames for comparison")

    # Render rows
    overlays_dir = output_dir / "sample_overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    categories = []
    titles = []
    for fid, cat, delta in selection:
        raw_s = raw_by_id[fid]
        ft_s = ft_by_id[fid]
        img_path = (dataset_dir / raw_s["image_path"]).resolve()
        if not img_path.exists():
            print(f"  skip {fid}: image not found")
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        panels = render_pair(img_bgr, raw_s, ft_s,
                             crop_size=args.crop_size, has_gt=has_gt,
                             align_2d=args.align_2d)
        rows.append((panels[0], panels[1]))
        categories.append(cat)

        if has_gt:
            unit = "px" if args.rank_by == "p2d" else "mm"
            title = (f"{fid}  "
                     f"raw {args.rank_by}={raw_s[args.rank_by]:.1f}{unit}  "
                     f"ft {args.rank_by}={ft_s[args.rank_by]:.1f}{unit}  "
                     f"(Δ={delta:+.1f}{unit})")
        else:
            title = fid
        titles.append(title)

        # Per-frame side-by-side JPEG
        side_by_side = np.concatenate(panels, axis=1)
        safe_id = fid.replace("/", "_")
        out_jpg = overlays_dir / f"{safe_id}_{cat}_compare.jpg"
        cv2.imwrite(str(out_jpg), side_by_side,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        print(f"  [{cat}] {fid}: Δ={delta:+.2f}mm -> {out_jpg.name}")

    grid_path = output_dir / "compare_grid.png"
    make_composite_grid(rows, categories, titles, grid_path,
                        dataset_name, has_gt, args.n_wins, args.n_losses,
                        args.rank_by, args.align_2d)


if __name__ == "__main__":
    main()
