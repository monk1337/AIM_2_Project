"""HaMeR model adapter, same crop-regress + GT bbox protocol as WiLoR.

Output convention (matches WiLoR adapter):
  pred_3d_mano:  (21, 3) MANO order, root-aligned (wrist ≈ 0)
  pred_2d_mano:  (21, 2) display frame
  pred_verts_mano: (778, 3)
  pred_cam_t_full: (3,) absolute translation in HaMeR's camera frame (uses HaMeR's focal=5000)
"""
import os
import sys
import argparse
import json
import time
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.chdir("/workspace/checkpoints/hamer")  # so DEFAULT_CHECKPOINT path resolves

sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from eval_joint_orders import mano_to_op, OP_VALID
from eval_metrics import metrics_3d, metrics_pve, metrics_2d, mrrpe_mm, aggregate
from eval_runner import compute_metrics, run_metrics_aggregation


def hamer_crop(img_np: np.ndarray, bbox_xywh: np.ndarray, is_right: int,
               img_size: int = 256, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Mirror HaMeR's ViTDetDataset crop. Returns normalized tensor (3, 256, 256) + box info."""
    import cv2
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2.0, y + h / 2.0
    bbox_size = max(w, h)
    flip = (is_right == 0)
    # Compute crop transform
    src_pts = np.array([[cx - bbox_size / 2, cy - bbox_size / 2],
                        [cx + bbox_size / 2, cy - bbox_size / 2],
                        [cx - bbox_size / 2, cy + bbox_size / 2]], dtype=np.float32)
    dst_pts = np.array([[0, 0], [img_size, 0], [0, img_size]], dtype=np.float32)
    M = cv2.getAffineTransform(src_pts, dst_pts)
    crop = cv2.warpAffine(img_np, M, (img_size, img_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if flip:
        crop = crop[:, ::-1].copy()
    img_t = crop.astype(np.float32) / 255.0
    img_t = (img_t - np.array(mean)) / np.array(std)
    img_t = np.transpose(img_t, (2, 0, 1)).astype(np.float32)
    return img_t, bbox_size, np.array([cx, cy], dtype=np.float32), flip


def cam_crop_to_full(pred_cam, box_center, box_size, img_size, focal_length):
    """HaMeR-style: convert weak-perspective crop cam to full-image translation.

    pred_cam: (B, 3) [s, tx, ty]
    img_size: (B, 2) [W, H]
    Returns t_full: (B, 3) translation in image frame using `focal_length`.
    """
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy = img_w / 2.0, img_h / 2.0
    s, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
    bx, by = box_center[:, 0], box_center[:, 1]
    tz = 2 * focal_length / (box_size * s + 1e-9)
    t_x_full = tx + (2 * (bx - cx) / (box_size * s + 1e-9))
    t_y_full = ty + (2 * (by - cy) / (box_size * s + 1e-9))
    return np.stack([t_x_full, t_y_full, tz], axis=1)


def perspective_project(j3d, t, fx, fy, cx, cy):
    """j3d: (N, 21, 3); t: (N, 3); returns (N, 21, 2)."""
    pts = j3d + t[:, None, :]
    u = fx * pts[..., 0] / np.maximum(pts[..., 2], 1e-6) + cx
    v = fy * pts[..., 1] / np.maximum(pts[..., 2], 1e-6) + cy
    return np.stack([u, v], axis=-1)


def run_hamer(rows, model, cfg, batch_size=64, device="cuda"):
    print("[crop] Building patches...")
    img_size = cfg.MODEL.IMAGE_SIZE
    mean = cfg.MODEL.IMAGE_MEAN
    std = cfg.MODEL.IMAGE_STD
    patches, meta = [], []
    for r in tqdm(rows, desc="crop"):
        img = np.asarray(get_image(r))
        bbox = derive_bbox_from_joints2d(r["native_joints_2d"], padding=1.5, img_wh=r["image_wh"])
        is_right = 1 if r["hand_side"] == "right" else 0
        patch, bsize, bcenter, flip = hamer_crop(img, bbox, is_right, img_size=img_size, mean=mean, std=std)
        patches.append(patch)
        meta.append({"row": r, "bbox_size": bsize, "box_center": bcenter, "flip": flip,
                     "is_right": is_right, "img_wh": r["image_wh"]})

    print("[fwd] Forward...")
    samples = []
    bs = batch_size
    f0 = cfg.EXTRA.FOCAL_LENGTH
    base_size = float(cfg.MODEL.IMAGE_SIZE)
    for i in tqdm(range(0, len(patches), bs), desc="forward"):
        batch_patches = np.stack(patches[i:i + bs])
        batch_meta = meta[i:i + bs]
        x = torch.from_numpy(batch_patches).to(device, dtype=torch.float32)
        with torch.no_grad():
            out = model({"img": x})
        out_np = {k: (v.cpu().float().numpy() if torch.is_tensor(v) else v) for k, v in out.items()}

        for j, m in enumerate(batch_meta):
            r = m["row"]
            pred_3d_mano = out_np["pred_keypoints_3d"][j].copy()    # (21, 3)
            pred_verts_mano = out_np["pred_vertices"][j].copy()      # (778, 3)
            pred_cam = out_np["pred_cam"][j:j+1].copy()              # (1, 3)
            if m["flip"]:
                pred_3d_mano[:, 0] = -pred_3d_mano[:, 0]
                pred_verts_mano[:, 0] = -pred_verts_mano[:, 0]
                pred_cam[:, 1] = -pred_cam[:, 1]
            img_size_arr = np.asarray(m["img_wh"], dtype=np.float32)
            scaled_focal = f0 / base_size * float(img_size_arr.max())
            pred_cam_t_full = cam_crop_to_full(pred_cam, m["box_center"][None],
                                               m["bbox_size"], img_size_arr[None], scaled_focal)
            pred_2d_mano = perspective_project(
                pred_3d_mano[None], pred_cam_t_full,
                fx=scaled_focal, fy=scaled_focal,
                cx=img_size_arr[0] / 2, cy=img_size_arr[1] / 2,
            )[0]
            samples.append({
                "row": r,
                "pred_3d_mano": pred_3d_mano,
                "pred_2d_mano": pred_2d_mano,
                "pred_verts_mano": pred_verts_mano,
                "pred_cam_t_full": pred_cam_t_full[0],
            })
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["aria_val", "pov_test"], default="pov_test")
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()
    if args.out is None:
        suffix = f"_stride{args.stride}" if args.dataset == "pov_test" and args.stride > 1 else ""
        args.out = f"/workspace/results/hamer_{args.dataset}{suffix}.json"

    print(f"[1/4] Loading {args.dataset}...")
    rows = load_aria_val() if args.dataset == "aria_val" else load_pov_test(stride=args.stride)
    if args.limit:
        rows = rows[:args.limit]
    print(f"  {len(rows)} instances")

    print("[2/4] Loading HaMeR...")
    from hamer.models import load_hamer
    model, cfg = load_hamer("/workspace/checkpoints/hamer/checkpoints/hamer.ckpt")
    model = model.cuda().eval()

    t0 = time.time()
    samples = run_hamer(rows, model, cfg, batch_size=args.batch_size)
    elapsed = time.time() - t0

    print(f"[3/4] Computing metrics ({args.dataset})...")
    samples = compute_metrics(samples, args.dataset, root_aligned_model=True)

    print("[4/4] Aggregating...")
    agg, per_sample = run_metrics_aggregation(samples, args.dataset, "HaMeR (off-the-shelf)", True)
    agg["elapsed_sec"] = elapsed

    print("\n=== OVERALL ===")
    keys = ["aria_native_pa_mpjpe_mm", "aria_native_p2d_px", "hsam_pa_mpjpe_mm", "hsam_pve_mm", "n"] \
        if args.dataset == "aria_val" else \
        ["pov_native_mpjpe_mm", "pov_native_pa_mpjpe_mm", "pov_native_pve_mm", "pov_native_pa_pve_mm", "pov_native_p2d_px", "n"]
    for k in keys:
        v = agg.get(k)
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== PER-VIDEO ===")
    pa_key = "aria_native_pa_mpjpe_mm" if args.dataset == "aria_val" else "pov_native_pa_mpjpe_mm"
    for seq, g in sorted(agg["per_group"].items()):
        print(f"  {seq}: PA={g.get(pa_key, float('nan')):.2f} mm  n={g['n']}")

    with open(args.out, "w") as f:
        json.dump({"summary": agg, "per_sample": per_sample}, f, default=float)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
