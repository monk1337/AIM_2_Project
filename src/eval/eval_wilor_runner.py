"""WiLoR runner, dual-GT + OP_VALID 20-joint subset for Aria + per-video + MRRPE.

Crop-regress mode using GT bbox derived from HaMeRSAM joints_2d (since WiLoR's
YOLO fails on Aria fisheye gloves).
"""
import sys
import argparse
import json
import time
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, "/workspace/code")
from eval_aria_loader import load_aria_val, derive_bbox_from_joints2d
from eval_joint_orders import mano_to_op, OP_VALID
from eval_metrics import metrics_3d, metrics_pve, metrics_2d, mrrpe_mm, aggregate


def crop_with_wilor(pipe, image_np: np.ndarray, bbox_xywh: np.ndarray, is_right: int):
    from wilor_mini.utils import utils
    import cv2
    from skimage.filters import gaussian

    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2.0, y + h / 2.0
    bbox_size = max(w, h)
    flip = (is_right == 0)
    patch = pipe.IMAGE_SIZE
    cvimg = image_np.copy()
    ds = (bbox_size / patch) / 2.0
    if ds > 1.1:
        cvimg = gaussian(cvimg, sigma=(ds - 1) / 2, channel_axis=2, preserve_range=True)
    img_patch_cv, _ = utils.generate_image_patch_cv2(
        cvimg, cx, cy, bbox_size, bbox_size, patch, patch,
        flip, 1.0, 0, border_mode=cv2.BORDER_CONSTANT)
    return img_patch_cv, bbox_size, np.array([cx, cy], dtype=np.float32), flip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/workspace/results/wilor_aria_val.json")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--rescale", type=float, default=2.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32")
    args = p.parse_args()

    print("[1/5] Loading WiLoR pipeline...")
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    from wilor_mini.utils import utils as wutils
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    pipe = WiLorHandPose3dEstimationPipeline(device=args.device, dtype=dtype, verbose=False)
    pipe.wilor_model.eval()

    print("[2/5] Loading Aria val...")
    rows = load_aria_val(only_eval_gt=True)
    if args.limit:
        rows = rows[:args.limit]
    print(f"  {len(rows)} instances")

    print("[3/5] Cropping...")
    patches, meta = [], []
    for r in tqdm(rows, desc="crop"):
        img = np.asarray(r["image"])
        bbox = derive_bbox_from_joints2d(r["hamersam_joints_2d_mano"], padding=args.rescale, img_wh=r["image_wh"])
        is_right = 1 if r["hand_side"] == "right" else 0
        patch, bbox_size, box_center, flip = crop_with_wilor(pipe, img, bbox, is_right)
        patches.append(patch)
        meta.append({"row": r, "bbox_size": bbox_size, "box_center": box_center, "flip": flip,
                     "is_right": is_right, "img_wh": r["image_wh"]})

    print("[4/5] Forward pass + per-sample metrics...")
    t0 = time.time()
    per_sample = []
    bs = args.batch_size
    for i in tqdm(range(0, len(patches), bs), desc="forward"):
        batch_patches = np.stack(patches[i:i + bs])
        batch_meta = meta[i:i + bs]
        x = torch.from_numpy(batch_patches).to(args.device, dtype=dtype)
        with torch.no_grad():
            out = pipe.wilor_model(x)
        out_np = {k: v.cpu().float().numpy() for k, v in out.items()}

        for j, m in enumerate(batch_meta):
            r = m["row"]
            wp = {k: v[[j]] for k, v in out_np.items()}
            pred_cam = wp["pred_cam"].copy()
            box_center = m["box_center"]
            bbox_size = m["bbox_size"]
            is_right = m["is_right"]
            flip = m["flip"]

            multiplier = (2 * is_right - 1)
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            pred_3d_mano = wp["pred_keypoints_3d"][0].copy()    # (21,3) MANO order
            pred_verts_mano = wp["pred_vertices"][0].copy()     # (778,3) MANO mesh
            if flip:
                pred_3d_mano[:, 0] = -pred_3d_mano[:, 0]
                pred_verts_mano[:, 0] = -pred_verts_mano[:, 0]
            pred_3d_op = mano_to_op(pred_3d_mano)              # (21,3) OP order

            img_size = np.asarray(m["img_wh"], dtype=np.float32)
            scaled_focal = pipe.FOCAL_LENGTH / pipe.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = wutils.cam_crop_to_full(
                pred_cam, box_center[None], bbox_size, img_size[None], scaled_focal)
            pred_2d_mano = wutils.perspective_projection(
                pred_3d_mano[None], translation=pred_cam_t_full,
                focal_length=np.array([scaled_focal] * 2)[None],
                camera_center=img_size[None] / 2,
            )[0]
            pred_2d_op = mano_to_op(pred_2d_mano)

            mm = {}
            # vs Aria native (PRIMARY, fair), OP_VALID 20-joint subset
            d3 = metrics_3d(pred_3d_op, r["eval_joints_3d_op"], joint_subset=OP_VALID)
            mm.update({f"aria_v2_{k}": v for k, v in d3.items()})
            d2 = metrics_2d(pred_2d_op, r["eval_joints_2d_op"], joint_subset=OP_VALID)
            mm.update({f"aria_v2_{k}": v for k, v in d2.items()})

            # vs HaMeRSAM pseudo-label (SECONDARY, circular for HaMeR-family), full 21 joints
            d3 = metrics_3d(pred_3d_mano, r["hamersam_joints_3d_mano"])
            mm.update({f"hsam_{k}": v for k, v in d3.items()})
            d2 = metrics_2d(pred_2d_mano, r["hamersam_joints_2d_mano"])
            mm.update({f"hsam_{k}": v for k, v in d2.items()})
            pve = metrics_pve(pred_verts_mano, r["hamersam_vertices"])
            mm.update({f"hsam_{k}": v for k, v in pve.items()})

            mm["sequence_name"] = r["sequence_name"]
            mm["frame_id"] = int(r["frame_id"])
            mm["hand_side"] = r["hand_side"]
            # Stash wrist preds for MRRPE (computed after loop)
            mm["_pred_wrist_op"] = pred_3d_op[0].tolist()
            mm["_gt_wrist_op_aria"] = r["eval_joints_3d_op"][0].tolist()
            mm["_gt_wrist_mano_hsam"] = r["hamersam_joints_3d_mano"][0].tolist()
            per_sample.append(mm)

    elapsed = time.time() - t0

    # MRRPE: pair right + left for same (sequence, frame), against both GTs
    print("[5/5] MRRPE on both-hand frames...")
    by_frame = defaultdict(dict)
    for s in per_sample:
        by_frame[(s["sequence_name"], s["frame_id"])][s["hand_side"]] = s
    n_pairs = 0
    mrrpe_aria_list, mrrpe_hsam_list = [], []
    for (seq, fid), pair in by_frame.items():
        if "right" in pair and "left" in pair:
            n_pairs += 1
            R, L = pair["right"], pair["left"]
            mrrpe_aria_list.append(mrrpe_mm(np.array(R["_pred_wrist_op"]), np.array(L["_pred_wrist_op"]),
                                            np.array(R["_gt_wrist_op_aria"]), np.array(L["_gt_wrist_op_aria"])))
            mrrpe_hsam_list.append(mrrpe_mm(np.array(R["_pred_wrist_op"]), np.array(L["_pred_wrist_op"]),
                                            np.array(R["_gt_wrist_mano_hsam"]), np.array(L["_gt_wrist_mano_hsam"])))
    # Strip private fields
    for s in per_sample:
        for k in list(s.keys()):
            if k.startswith("_"):
                del s[k]

    agg = aggregate(per_sample, group_key="sequence_name")
    agg["aria_mrrpe_mm"] = float(np.mean(mrrpe_aria_list)) if mrrpe_aria_list else None
    agg["hsam_mrrpe_mm"] = float(np.mean(mrrpe_hsam_list)) if mrrpe_hsam_list else None
    agg["mrrpe_n_pairs"] = n_pairs
    agg["model"] = "WiLoR (off-the-shelf, GT-bbox crop-regress)"
    agg["dataset"] = "Aria val (PR81-85)"
    agg["elapsed_sec"] = elapsed

    print("\n=== OVERALL ===")
    keys_to_show = [
        "aria_v2_mpjpe_mm", "aria_v2_pa_mpjpe_mm", "aria_v2_p2d_px",
        "hsam_mpjpe_mm", "hsam_pa_mpjpe_mm", "hsam_pve_mm", "hsam_pa_pve_mm", "hsam_p2d_px",
        "aria_mrrpe_mm", "hsam_mrrpe_mm", "mrrpe_n_pairs", "n",
    ]
    for k in keys_to_show:
        v = agg.get(k)
        if v is not None:
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")

    print("\n=== PER-VIDEO (PA-MPJPE_v2 vs Aria | PA-MPJPE vs HSAM | n) ===")
    for seq, g in sorted(agg["per_group"].items()):
        print(f"  {seq}: aria_PA={g.get('aria_v2_pa_mpjpe_mm', float('nan')):.2f} mm  "
              f"hsam_PA={g.get('hsam_pa_mpjpe_mm', float('nan')):.2f} mm  "
              f"aria_p2d={g.get('aria_v2_p2d_px', float('nan')):.1f} px  n={g['n']}")

    with open(args.out, "w") as f:
        json.dump({"summary": agg, "per_sample": per_sample}, f, default=float)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
