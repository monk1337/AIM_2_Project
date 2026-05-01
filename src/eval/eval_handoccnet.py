"""HandOccNet adapter, outputs MANO joints + verts in MANO order, root-aligned.

Repo expects to run from main/ dir; we'll set sys.path manually.
"""
import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import cv2
from tqdm import tqdm

# Wire HandOccNet's import paths, local manopth must come BEFORE site-packages
HONET = "/workspace/code/HandOccNet"
sys.path.insert(0, f"{HONET}/main")
sys.path.insert(0, f"{HONET}/common")
sys.path.insert(0, f"{HONET}/common/utils/manopth")  # custom manopth that returns 3-tuple

# Override mano_path BEFORE imports (the cfg uses it on init)
os.environ["MANO_DIR"] = "/workspace/mano"

sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from eval_metrics import metrics_3d, metrics_pve, metrics_2d
from eval_runner import compute_metrics, run_metrics_aggregation


def honet_load_model():
    from config import cfg
    cfg.set_args("0")
    # Patch mano_path: HandOccNet uses cfg.mano_path/mano/models/MANO_RIGHT.pkl
    # We have /workspace/mano/MANO_RIGHT.pkl, so create a symlink structure
    target_dir = "/workspace/checkpoints/handoccnet/_mano_root/mano/models"
    os.makedirs(target_dir, exist_ok=True)
    src = "/workspace/mano/MANO_RIGHT.pkl"
    dst = f"{target_dir}/MANO_RIGHT.pkl"
    if not os.path.exists(dst):
        os.symlink(src, dst)
    cfg.mano_path = "/workspace/checkpoints/handoccnet/_mano_root"

    from model import get_model
    model = get_model("test")
    from torch.nn.parallel.data_parallel import DataParallel
    model = DataParallel(model).cuda()
    ckpt = torch.load("/workspace/checkpoints/handoccnet/HandOccNet_model_dump/snapshot_demo.pth.tar",
                      map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["network"], strict=False)
    model.eval()
    return model


def honet_crop(img_np, bbox_xywh):
    """HandOccNet's crop: 256x256, no flip handling needed (HandOccNet handles single-side)."""
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2.0, y + h / 2.0
    bbox_size = max(w, h)
    src_pts = np.array([[cx - bbox_size / 2, cy - bbox_size / 2],
                        [cx + bbox_size / 2, cy - bbox_size / 2],
                        [cx - bbox_size / 2, cy + bbox_size / 2]], dtype=np.float32)
    dst_pts = np.array([[0, 0], [256, 0], [0, 256]], dtype=np.float32)
    M = cv2.getAffineTransform(src_pts, dst_pts)
    crop = cv2.warpAffine(img_np, M, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    img_t = crop.astype(np.float32) / 255.0
    img_t = np.transpose(img_t, (2, 0, 1))
    return img_t, bbox_size, np.array([cx, cy], dtype=np.float32)


def run_honet(rows, model, batch_size=64, device="cuda"):
    print("[crop] Building patches...")
    patches, meta = [], []
    for r in tqdm(rows, desc="crop"):
        img = np.asarray(get_image(r))
        bbox = derive_bbox_from_joints2d(r["native_joints_2d"], padding=1.5, img_wh=r["image_wh"])
        is_right = 1 if r["hand_side"] == "right" else 0
        # HandOccNet trained on right hands; flip left hands
        if not is_right:
            img = img[:, ::-1].copy()
            bbox = bbox.copy()
            bbox[0] = img.shape[1] - bbox[0] - bbox[2]
        patch, bsize, bcenter = honet_crop(img, bbox)
        if not is_right:
            # store flip flag for un-flip
            pass
        patches.append(patch)
        meta.append({"row": r, "bbox_size": bsize, "box_center": bcenter, "is_right": is_right,
                     "img_wh": r["image_wh"], "flip": (not is_right)})

    print("[fwd] Forward...")
    samples = []
    bs = batch_size
    for i in tqdm(range(0, len(patches), bs), desc="forward"):
        batch_patches = np.stack(patches[i:i + bs])
        batch_meta = meta[i:i + bs]
        x = torch.from_numpy(batch_patches).to(device, dtype=torch.float32)
        with torch.no_grad():
            out = model({"img": x}, {}, {}, "test")
        joints_np = out["joints_coord_cam"].cpu().float().numpy()  # (B, 21, 3) IN OP ORDER, root-aligned, meters
        verts_np = out["mesh_coord_cam"].cpu().float().numpy()      # (B, 778, 3)

        # Keep in OP order; runner uses pred_in_op_order=True
        for j, m in enumerate(batch_meta):
            r = m["row"]
            pred_3d_mano = joints_np[j].copy()  # despite name, this is OP order
            pred_verts_mano = verts_np[j].copy()
            if m["flip"]:
                pred_3d_mano[:, 0] = -pred_3d_mano[:, 0]
                pred_verts_mano[:, 0] = -pred_verts_mano[:, 0]
            # No 2D output, use placeholder 0s; P2D will be huge but flagged
            pred_2d_mano = np.zeros((21, 2), dtype=np.float32)
            samples.append({
                "row": r,
                "pred_3d_mano": pred_3d_mano,
                "pred_2d_mano": pred_2d_mano,
                "pred_verts_mano": pred_verts_mano,
                "pred_cam_t_full": np.zeros(3, dtype=np.float32),
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
        args.out = f"/workspace/results/handoccnet_{args.dataset}{suffix}.json"

    print(f"[1/4] Loading {args.dataset}...")
    rows = load_aria_val() if args.dataset == "aria_val" else load_pov_test(stride=args.stride)
    if args.limit:
        rows = rows[:args.limit]
    print(f"  {len(rows)} instances")

    print("[2/4] Loading HandOccNet...")
    model = honet_load_model()

    t0 = time.time()
    samples = run_honet(rows, model, batch_size=args.batch_size)
    elapsed = time.time() - t0

    print(f"[3/4] Computing metrics ({args.dataset})...")
    samples = compute_metrics(samples, args.dataset, root_aligned_model=True, pred_in_op_order=True)

    print("[4/4] Aggregating...")
    agg, per_sample = run_metrics_aggregation(samples, args.dataset, "HandOccNet (off-the-shelf)", True)
    agg["elapsed_sec"] = elapsed

    print("\n=== OVERALL ===")
    keys = ["aria_native_pa_mpjpe_mm", "hsam_pa_mpjpe_mm", "hsam_pve_mm", "n"] if args.dataset == "aria_val" else \
           ["pov_native_mpjpe_mm", "pov_native_pa_mpjpe_mm", "pov_native_pve_mm", "pov_native_pa_pve_mm", "n"]
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
