"""Deep debug of Aria training data feeding.

Compares:
1. POV train sample: image + GT 3D + WiLoR off-shelf pred 3D → should be reasonable
2. Aria train sample: image + GT 3D + WiLoR off-shelf pred 3D → expected to match HSAM closely

Saves visualizations + prints joint magnitudes/orders for inspection.
"""
import os
import sys
import io
import numpy as np
import torch
import cv2
import json
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

sys.path.insert(0, "/workspace/code")
from eval_loader import derive_bbox_from_joints2d
from eval_joint_orders import mano_to_op


def load_pov_sample(idx=0):
    files = sorted(Path("/workspace/datasets/pov_surgery/data").glob("train-*.parquet"))
    df = pq.read_table(files[0]).to_pandas()
    r = df.iloc[idx]
    return {
        "ds": "pov",
        "seq": r.sequence_name, "fid": int(r.frame_id), "side": r.hand_side,
        "img_pil": Image.open(io.BytesIO(r.image["bytes"])).convert("RGB"),
        "img_wh": (int(r.image_width), int(r.image_height)),
        "joints_2d": np.asarray(r.joints_2d, dtype=np.float32).reshape(21, 2),
        "joints_3d": np.asarray(r.joints_3d, dtype=np.float32).reshape(21, 3),
        "vertices": np.asarray(r.vertices, dtype=np.float32).reshape(778, 3),
    }


def load_aria_sample(idx=0):
    files = sorted(Path("/workspace/datasets/aria_val/data").glob("train-*.parquet"))
    df = pq.read_table(files[0]).to_pandas()
    r = df.iloc[idx]
    W = int(r.image_width)
    j2d_raw = np.asarray(r.joints_2d, dtype=np.float32).reshape(21, 2)
    j2d_disp = np.stack([j2d_raw[:, 1], (W - 1) - j2d_raw[:, 0]], axis=-1).astype(np.float32)  # CCW90
    return {
        "ds": "aria",
        "seq": r.sequence_name, "fid": int(r.frame_id), "side": r.hand_side,
        "img_pil": Image.open(io.BytesIO(r.image["bytes"])).convert("RGB"),
        "img_wh": (W, int(r.image_height)),
        "joints_2d_raw": j2d_raw,
        "joints_2d": j2d_disp,
        "joints_3d": np.asarray(r.joints_3d, dtype=np.float32).reshape(21, 3),
        "vertices": np.asarray(r.vertices, dtype=np.float32).reshape(778, 3),
    }


def crop_for_wilor(img_np, joints_2d, hand_side, image_size=256):
    bbox = derive_bbox_from_joints2d(joints_2d, padding=1.5, img_wh=(img_np.shape[1], img_np.shape[0]))
    is_right = 1 if hand_side == "right" else 0
    flip = (is_right == 0)
    x_, y_, w_, h_ = bbox
    cx, cy = x_ + w_ / 2, y_ + h_ / 2
    bsize = max(w_, h_)
    if flip:
        img_np = img_np[:, ::-1].copy()
        cx = img_np.shape[1] - cx
    src = np.array([[cx - bsize / 2, cy - bsize / 2],
                    [cx + bsize / 2, cy - bsize / 2],
                    [cx - bsize / 2, cy + bsize / 2]], dtype=np.float32)
    dst = np.array([[0, 0], [image_size, 0], [0, image_size]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(img_np, M, (image_size, image_size), flags=cv2.INTER_LINEAR)
    return crop, flip, M, bbox


def run_wilor(crop_rgb, pipe):
    """crop_rgb is (H, W, 3) RGB uint8. wilor expects BGR (H, W, 3). Apply flip(-1) internally."""
    crop_bgr = crop_rgb[:, :, ::-1].copy().astype(np.float32)
    x = torch.from_numpy(crop_bgr[None]).to("cuda", dtype=torch.float32)
    with torch.no_grad():
        out = pipe.wilor_model(x)
    return {k: v[0].cpu().float().numpy() for k, v in out.items() if torch.is_tensor(v)}


def main():
    print("[1/4] Load WiLoR pretrained...")
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    pipe = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
    pipe.wilor_model.eval()

    out_dir = "/tmp/aria_debug"
    os.makedirs(out_dir, exist_ok=True)

    for source in ["pov", "aria"]:
        for idx in [0, 100, 500, 1000]:
            print(f"\n=== {source} idx={idx} ===")
            samp = load_pov_sample(idx) if source == "pov" else load_aria_sample(idx)
            img_np = np.asarray(samp["img_pil"])
            print(f"  seq={samp['seq']}  fid={samp['fid']}  side={samp['side']}  img_wh={samp['img_wh']}")
            print(f"  GT joints_2d range: x=[{samp['joints_2d'][:,0].min():.0f}, {samp['joints_2d'][:,0].max():.0f}]  y=[{samp['joints_2d'][:,1].min():.0f}, {samp['joints_2d'][:,1].max():.0f}]")
            print(f"  GT joints_3d wrist: {samp['joints_3d'][0]}")
            print(f"  GT joints_3d shape (root-rel), extent (max ‖xy‖): {np.linalg.norm((samp['joints_3d'] - samp['joints_3d'][0:1])[:, :2], axis=-1).max():.4f} m")
            print(f"  GT joints_3d depth range (z): [{samp['joints_3d'][:, 2].min():.3f}, {samp['joints_3d'][:, 2].max():.3f}]")

            crop, flip, M, bbox = crop_for_wilor(img_np, samp["joints_2d"], samp["side"])
            print(f"  bbox xywh: {bbox}  flip={flip}")
            # save crop with GT 2D overlaid
            crop_pil = Image.fromarray(crop)
            d = ImageDraw.Draw(crop_pil)
            # transform GT joints_2d to crop coords
            j2d_for_crop = samp["joints_2d"].copy()
            if flip:
                j2d_for_crop[:, 0] = img_np.shape[1] - j2d_for_crop[:, 0]
            j2d_h = np.concatenate([j2d_for_crop, np.ones((21, 1))], axis=1)
            j2d_in_crop = (M @ j2d_h.T).T  # (21, 2)
            for k, (x, y) in enumerate(j2d_in_crop):
                d.ellipse([x-3, y-3, x+3, y+3], fill=(0, 255, 0), outline=(0, 0, 0))
            crop_pil.save(f"{out_dir}/{source}_{idx}_crop_gt.jpg", quality=85)

            # Run WiLoR
            out = run_wilor(crop, pipe)
            pred_3d = out["pred_keypoints_3d"]
            if flip:
                pred_3d = pred_3d.copy()
                pred_3d[:, 0] = -pred_3d[:, 0]
            pred_3d_root = pred_3d - pred_3d[0:1]
            gt_3d_root = samp["joints_3d"] - samp["joints_3d"][0:1]
            print(f"  WiLoR pred_3d wrist: {pred_3d[0]}")
            print(f"  WiLoR pred shape extent: {np.linalg.norm(pred_3d_root[:, :2], axis=-1).max():.4f} m")
            # Per-joint L1 (root-aligned)
            per_joint_l1 = np.linalg.norm(pred_3d_root - gt_3d_root, axis=-1)
            print(f"  per-joint L1 (root-rel), mean: {per_joint_l1.mean()*1000:.2f} mm  max: {per_joint_l1.max()*1000:.2f} mm")
            # Procrustes-aligned MPJPE
            from eval_metrics import procrustes_align
            aligned = procrustes_align(pred_3d_root, gt_3d_root)[0]
            pa_err = np.linalg.norm(aligned - gt_3d_root, axis=-1).mean() * 1000
            print(f"  PA-MPJPE: {pa_err:.2f} mm")

    print(f"\nSaved crops to {out_dir}/")


if __name__ == "__main__":
    main()
