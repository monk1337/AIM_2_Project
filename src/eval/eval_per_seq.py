"""Per-sequence Aria val PA-MPJPE: off-shelf vs FT'd WiLoR.

For each Aria sequence, compute mean PA-MPJPE per model. Print sorted by
improvement (off - ft). Used to pick the best sequences for mesh-overlay viz.
"""
import sys
import os
import argparse
import json
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from eval_joint_orders import MANO_TO_OPENPOSE
from eval_metrics import metrics_3d
from torch.amp import autocast
import cv2
from tqdm import tqdm

OP_VALID = [0] + list(range(2, 21))


def crop_for_wilor(img, bbox, hand_side, image_size=256):
    is_right = 1 if hand_side == "right" else 0
    flip = (is_right == 0)
    if flip:
        img = img[:, ::-1].copy()
        bbox = bbox.copy()
        bbox[0] = img.shape[1] - bbox[0] - bbox[2]
    x_, y_, w_, h_ = bbox
    cx, cy = x_ + w_ / 2, y_ + h_ / 2
    bsize = max(w_, h_)
    src = np.array([[cx - bsize / 2, cy - bsize / 2],
                    [cx + bsize / 2, cy - bsize / 2],
                    [cx - bsize / 2, cy + bsize / 2]], dtype=np.float32)
    dst = np.array([[0, 0], [image_size, 0], [0, image_size]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, M, (image_size, image_size), flags=cv2.INTER_LINEAR)


def run_model_on_samples(model, samples, kind, batch_size=64):
    """Returns list of dicts with sample_idx, sequence_name, and PA-MPJPE
    against each available GT (mps, hsam, native).
    """
    out = []
    n = len(samples)
    for start in tqdm(range(0, n, batch_size), desc=kind):
        end = min(start + batch_size, n)
        batch = samples[start:end]
        crops, sides_flipped = [], []
        for samp in batch:
            img = np.asarray(get_image(samp))
            bbox = derive_bbox_from_joints2d(samp["native_joints_2d"], padding=1.5,
                                             img_wh=samp["image_wh"])
            crop = crop_for_wilor(img, bbox, samp["hand_side"])
            crops.append(crop)
            sides_flipped.append(samp["hand_side"] != "right")
        crops_bgr = np.stack([c[:, :, ::-1].astype(np.float32) for c in crops])
        x = torch.from_numpy(crops_bgr).cuda().float()
        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            res = model(x)
        kp_batch = res["pred_keypoints_3d"].float().cpu().numpy()
        for j, samp in enumerate(batch):
            kp = kp_batch[j]
            if sides_flipped[j]:
                kp = kp.copy()
                kp[:, 0] = -kp[:, 0]
            pred_op = kp[MANO_TO_OPENPOSE]
            row = {"idx": start + j, "seq": samp["sequence_name"]}
            if kind == "aria":
                gt_mps = samp.get("aria_eval_joints_3d_op")
                if gt_mps is not None and np.isfinite(gt_mps).all():
                    row["mps_mm"] = metrics_3d(pred_op, gt_mps, joint_subset=OP_VALID)["pa_mpjpe_mm"]
                gt_hsam = samp["native_joints_3d"][MANO_TO_OPENPOSE]
                row["hsam_mm"] = metrics_3d(pred_op, gt_hsam)["pa_mpjpe_mm"]
            else:
                gt_op = samp["native_joints_3d"][MANO_TO_OPENPOSE]
                row["pov_mm"] = metrics_3d(pred_op, gt_op)["pa_mpjpe_mm"]
            out.append(row)
    return out


def aggregate_by_seq(rows, key):
    by = defaultdict(list)
    for r in rows:
        if key in r:
            by[r["seq"]].append(r[key])
    return {seq: (float(np.mean(v)), len(v)) for seq, v in by.items() if v}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ft_ckpt", required=True)
    p.add_argument("--out_json", default="/workspace/results/per_seq_eval.json")
    args = p.parse_args()

    print("[1/4] Loading models...")
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    pipe_off = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
    model_off = pipe_off.wilor_model.eval()
    pipe_ft = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
    ck = torch.load(args.ft_ckpt, map_location="cuda", weights_only=False)
    pipe_ft.wilor_model.load_state_dict(ck["model_state_dict"])
    model_ft = pipe_ft.wilor_model.eval()

    print("[2/4] Loading samples...")
    aria = load_aria_val()
    pov = load_pov_test(stride=1)
    print(f"  Aria n={len(aria)}, POV n={len(pov)}")

    print("[3/4] Eval off-shelf...")
    aria_off = run_model_on_samples(model_off, aria, "aria")
    pov_off = run_model_on_samples(model_off, pov, "pov")
    print("[3/4] Eval FT...")
    aria_ft = run_model_on_samples(model_ft, aria, "aria")
    pov_ft = run_model_on_samples(model_ft, pov, "pov")

    rows = []
    # Aria: report both HSAM (training target) and MPS (held-out gold)
    for metric in ("hsam_mm", "mps_mm"):
        off_by = aggregate_by_seq(aria_off, metric)
        ft_by = aggregate_by_seq(aria_ft, metric)
        for seq in sorted(set(off_by) | set(ft_by)):
            o = off_by.get(seq, (float("nan"), 0))
            f = ft_by.get(seq, (float("nan"), 0))
            rows.append({"dataset": "aria", "metric": metric, "seq": seq, "n": o[1],
                         "off_mm": o[0], "ft_mm": f[0], "delta_mm": o[0] - f[0]})
    pov_off_by = aggregate_by_seq(pov_off, "pov_mm")
    pov_ft_by = aggregate_by_seq(pov_ft, "pov_mm")
    for seq in sorted(set(pov_off_by) | set(pov_ft_by)):
        o = pov_off_by.get(seq, (float("nan"), 0))
        f = pov_ft_by.get(seq, (float("nan"), 0))
        rows.append({"dataset": "pov", "metric": "pov_mm", "seq": seq, "n": o[1],
                     "off_mm": o[0], "ft_mm": f[0], "delta_mm": o[0] - f[0]})

    print("\n=== Aria HSAM (training target), sorted by improvement ===")
    for r in sorted([x for x in rows if x["dataset"] == "aria" and x["metric"] == "hsam_mm"],
                    key=lambda r: -r["delta_mm"]):
        print(f"  {r['seq']:30s} n={r['n']:4d}  off={r['off_mm']:6.2f}  "
              f"ft={r['ft_mm']:6.2f}  Δ={r['delta_mm']:+6.2f}")
    print("\n=== Aria MPS (held-out gold) ===")
    for r in sorted([x for x in rows if x["dataset"] == "aria" and x["metric"] == "mps_mm"],
                    key=lambda r: -r["delta_mm"]):
        print(f"  {r['seq']:30s} n={r['n']:4d}  off={r['off_mm']:6.2f}  "
              f"ft={r['ft_mm']:6.2f}  Δ={r['delta_mm']:+6.2f}")
    print("\n=== POV (top wins) ===")
    for r in sorted([x for x in rows if x["dataset"] == "pov"],
                    key=lambda r: -r["delta_mm"])[:8]:
        print(f"  {r['seq']:30s} n={r['n']:4d}  off={r['off_mm']:6.2f}  "
              f"ft={r['ft_mm']:6.2f}  Δ={r['delta_mm']:+6.2f}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as fp:
        json.dump({
            "aria_off": aria_off, "aria_ft": aria_ft,
            "pov_off": pov_off, "pov_ft": pov_ft,
            "per_seq": rows,
        }, fp, indent=2, default=float)
    print(f"\nSaved {args.out_json}")


if __name__ == "__main__":
    main()
