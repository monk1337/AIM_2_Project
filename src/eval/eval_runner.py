"""Unified runner for any model on any dataset (Aria val / POV test).

Per-sample metrics:
  vs PRIMARY GT:
    - Aria val: aria_native_* (Aria MPS, OP_VALID 20-joint subset)
    - POV test: pov_native_*  (synthetic MANO, all 21 joints)

  vs SECONDARY GT (Aria val only, HaMeRSAM pseudo, marked circular for HaMeR-family):
    - hsam_* (full 21 joints, MANO order)

Per-video aggregates + overall + MRRPE on both-hand frames.
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
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from eval_joint_orders import mano_to_op, OP_VALID
from eval_metrics import metrics_3d, metrics_pve, metrics_2d, mrrpe_mm, aggregate


def crop_with_wilor(pipe, image_np, bbox_xywh, is_right):
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


def run_wilor(rows, pipe, batch_size=128, dtype=torch.float32, device="cuda"):
    """Run WiLoR on a list of dataset rows. Returns list of per-sample dicts with predictions + metrics."""
    from wilor_mini.utils import utils as wutils

    print("[crop] Building patches (canonical bbox: 1.5× clamped [260,1200])...")
    patches, meta = [], []
    for r in tqdm(rows, desc="crop"):
        img = np.asarray(get_image(r))
        bbox = derive_bbox_from_joints2d(r["native_joints_2d"], padding=1.5, img_wh=r["image_wh"])
        is_right = 1 if r["hand_side"] == "right" else 0
        patch, bbox_size, box_center, flip = crop_with_wilor(pipe, img, bbox, is_right)
        patches.append(patch)
        meta.append({"row": r, "bbox_size": bbox_size, "box_center": box_center, "flip": flip,
                     "is_right": is_right, "img_wh": r["image_wh"]})

    print("[fwd] Forward...")
    per_sample = []
    bs = batch_size
    pipe.wilor_model.eval()
    for i in tqdm(range(0, len(patches), bs), desc="forward"):
        batch_patches = np.stack(patches[i:i + bs])
        batch_meta = meta[i:i + bs]
        x = torch.from_numpy(batch_patches).to(device, dtype=dtype)
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
            pred_3d_mano = wp["pred_keypoints_3d"][0].copy()
            pred_verts_mano = wp["pred_vertices"][0].copy()
            if flip:
                pred_3d_mano[:, 0] = -pred_3d_mano[:, 0]
                pred_verts_mano[:, 0] = -pred_verts_mano[:, 0]

            img_size = np.asarray(m["img_wh"], dtype=np.float32)
            scaled_focal = pipe.FOCAL_LENGTH / pipe.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = wutils.cam_crop_to_full(
                pred_cam, box_center[None], bbox_size, img_size[None], scaled_focal)
            pred_2d_mano = wutils.perspective_projection(
                pred_3d_mano[None], translation=pred_cam_t_full,
                focal_length=np.array([scaled_focal] * 2)[None],
                camera_center=img_size[None] / 2,
            )[0]

            sample = {
                "row": r,
                "pred_3d_mano": pred_3d_mano,
                "pred_2d_mano": pred_2d_mano,
                "pred_verts_mano": pred_verts_mano,
                "pred_cam_t_full": pred_cam_t_full[0],  # (3,) absolute camera-frame translation
            }
            per_sample.append(sample)
    return per_sample


def compute_metrics(samples: list, dataset: str, root_aligned_model: bool = True,
                    pred_in_op_order: bool = False):
    """Compute per-sample metrics. Adds dict keys to each sample.

    pred_in_op_order=True → pred_3d_mano is actually in OP order already (e.g. MeshGraphormer/HandOccNet).
                            Skip the MANO→OP permutation for Aria. For POV, permute GT MANO→OP for comparison.
    pred_in_op_order=False (default) → pred_3d_mano is in standard MANO order (HaMeR/WiLoR).
    """
    drop_abs_keys = ("mpjpe_abs_mm", "root_err_mm") if root_aligned_model else ()

    def filter_abs(d: dict) -> dict:
        return {k: v for k, v in d.items() if k not in drop_abs_keys}

    for s in samples:
        r = s["row"]
        pred_3d_mano = s["pred_3d_mano"]   # may actually be OP order if pred_in_op_order=True
        pred_2d_mano = s["pred_2d_mano"]
        pred_verts_mano = s["pred_verts_mano"]
        pred_3d_op = pred_3d_mano if pred_in_op_order else mano_to_op(pred_3d_mano)

        mm = {}
        if dataset == "aria_val":
            d3 = metrics_3d(pred_3d_op, r["aria_eval_joints_3d_op"], joint_subset=OP_VALID)
            mm.update({f"aria_native_{k}": v for k, v in filter_abs(d3).items()})
            mps_2d = r.get("aria_mps_2d_disp")
            if mps_2d is not None:
                d2 = metrics_2d(pred_2d_mano, mps_2d)
                mm.update({f"aria_native_{k}": v for k, v in d2.items()})

            # HSAM joints, for OP-order preds, convert HSAM MANO → OP for fair comparison
            if not pred_in_op_order:
                d3 = metrics_3d(pred_3d_mano, r["native_joints_3d"])
                mm.update({f"hsam_{k}": v for k, v in filter_abs(d3).items()})
                d2 = metrics_2d(pred_2d_mano, r["native_joints_2d"])
                mm.update({f"hsam_{k}": v for k, v in d2.items()})
            else:
                hsam_3d_op = mano_to_op(r["native_joints_3d"])
                d3 = metrics_3d(pred_3d_op, hsam_3d_op)
                mm.update({f"hsam_{k}": v for k, v in filter_abs(d3).items()})
                # 2D: HSAM joints_2d is in MANO order, convert to OP for comparison
                hsam_2d_op = mano_to_op(r["native_joints_2d"])
                d2 = metrics_2d(pred_2d_mano, hsam_2d_op)
                mm.update({f"hsam_{k}": v for k, v in d2.items()})
            # PVE shape error, order-invariant via Procrustes
            pve = metrics_pve(pred_verts_mano, r["native_vertices"])
            mm.update({f"hsam_{k}": v for k, v in pve.items()})
        else:  # pov_test, POV native_joints_3d is in standard MANO order
            if pred_in_op_order:
                # Convert POV GT to OP order for fair comparison
                pov_gt_op = mano_to_op(r["native_joints_3d"])
                d3 = metrics_3d(pred_3d_op, pov_gt_op)
            else:
                d3 = metrics_3d(pred_3d_mano, r["native_joints_3d"])
            mm.update({f"pov_native_{k}": v for k, v in filter_abs(d3).items()})
            d2 = metrics_2d(pred_2d_mano, r["native_joints_2d"])
            mm.update({f"pov_native_{k}": v for k, v in d2.items()})
            pve = metrics_pve(pred_verts_mano, r["native_vertices"])
            mm.update({f"pov_native_{k}": v for k, v in pve.items()})

        mm["sequence_name"] = r["sequence_name"]
        mm["frame_id"] = int(r["frame_id"])
        mm["hand_side"] = r["hand_side"]
        if not root_aligned_model:
            mm["_pred_wrist_mano"] = pred_3d_mano[0].tolist()
            if dataset == "aria_val":
                mm["_gt_wrist_aria_op"] = r["aria_eval_joints_3d_op"][0].tolist()
            mm["_gt_wrist_native"] = r["native_joints_3d"][0].tolist()
        s["metrics"] = mm
    return samples


def run_metrics_aggregation(samples, dataset, model_name, root_aligned_model):
    per_sample = [s["metrics"] for s in samples]

    by_frame = defaultdict(dict)
    for s in per_sample:
        by_frame[(s["sequence_name"], s["frame_id"])][s["hand_side"]] = s
    n_pairs = sum(1 for pair in by_frame.values() if "right" in pair and "left" in pair)

    mrrpe_native, mrrpe_aria = None, None
    if not root_aligned_model:
        mrrpe_native_list, mrrpe_aria_list = [], []
        for pair in by_frame.values():
            if "right" not in pair or "left" not in pair:
                continue
            R, L = pair["right"], pair["left"]
            mrrpe_native_list.append(mrrpe_mm(
                np.array(R["_pred_wrist_mano"]), np.array(L["_pred_wrist_mano"]),
                np.array(R["_gt_wrist_native"]), np.array(L["_gt_wrist_native"])))
            if dataset == "aria_val":
                mrrpe_aria_list.append(mrrpe_mm(
                    np.array(R["_pred_wrist_mano"]), np.array(L["_pred_wrist_mano"]),
                    np.array(R["_gt_wrist_aria_op"]), np.array(L["_gt_wrist_aria_op"])))
        mrrpe_native = float(np.mean(mrrpe_native_list)) if mrrpe_native_list else None
        if dataset == "aria_val":
            mrrpe_aria = float(np.mean(mrrpe_aria_list)) if mrrpe_aria_list else None

    # Strip private fields
    for s in per_sample:
        for k in list(s.keys()):
            if k.startswith("_"):
                del s[k]

    agg = aggregate(per_sample, group_key="sequence_name")
    agg["mrrpe_n_pairs"] = n_pairs
    if root_aligned_model:
        agg["mrrpe_native_mm"] = "N/A (root-aligned model)"
        if dataset == "aria_val":
            agg["mrrpe_aria_mm"] = "N/A (root-aligned model)"
        agg["mpjpe_abs_root_err_note"] = "Root err and MPJPE-abs not computed (model outputs root-aligned predictions)."
    else:
        agg["mrrpe_native_mm"] = mrrpe_native
        if dataset == "aria_val":
            agg["mrrpe_aria_mm"] = mrrpe_aria
    agg["model"] = model_name
    agg["dataset"] = dataset
    return agg, per_sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="wilor", choices=["wilor"])  # more models added later
    p.add_argument("--dataset", default="aria_val", choices=["aria_val", "pov_test"])
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--stride", type=int, default=1, help="POV: take every Nth frame per sequence")
    args = p.parse_args()
    if args.out is None:
        suffix = f"_stride{args.stride}" if args.dataset == "pov_test" and args.stride > 1 else ""
        args.out = f"/workspace/results/{args.model}_{args.dataset}{suffix}.json"

    print(f"[1/4] Loading {args.dataset}...")
    rows = load_aria_val() if args.dataset == "aria_val" else load_pov_test(stride=args.stride)
    if args.limit:
        rows = rows[:args.limit]
    print(f"  {len(rows)} instances")

    print(f"[2/4] Loading {args.model}...")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if args.model == "wilor":
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
        pipe = WiLorHandPose3dEstimationPipeline(device=args.device, dtype=dtype, verbose=False)
        root_aligned = True  # WiLoR pred_keypoints_3d is in MANO-local frame (wrist ≈ 0)
        model_name = "WiLoR (off-the-shelf, GT-bbox crop-regress)"
        t0 = time.time()
        samples = run_wilor(rows, pipe, batch_size=args.batch_size, dtype=dtype, device=args.device)
        elapsed = time.time() - t0
    else:
        raise ValueError(f"Unknown model {args.model}")

    print(f"[3/4] Computing metrics ({args.dataset})...")
    samples = compute_metrics(samples, args.dataset, root_aligned_model=root_aligned)

    print("[4/4] Aggregating...")
    agg, per_sample = run_metrics_aggregation(samples, args.dataset, model_name, root_aligned)
    agg["elapsed_sec"] = elapsed

    # Print summary
    print("\n=== OVERALL ===")
    if args.dataset == "aria_val":
        show_keys = ["aria_native_mpjpe_mm", "aria_native_pa_mpjpe_mm", "aria_native_p2d_px",
                     "hsam_mpjpe_mm", "hsam_pa_mpjpe_mm",
                     "hsam_pve_mm", "hsam_pa_pve_mm", "hsam_p2d_px",
                     "mrrpe_aria_mm", "mrrpe_native_mm", "mrrpe_n_pairs", "n"]
    else:
        show_keys = ["pov_native_mpjpe_mm", "pov_native_pa_mpjpe_mm",
                     "pov_native_pve_mm", "pov_native_pa_pve_mm", "pov_native_p2d_px",
                     "mrrpe_native_mm", "mrrpe_n_pairs", "n"]
    for k in show_keys:
        v = agg.get(k)
        if v is None:
            print(f"  {k}: N/A")
        elif isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    print("\n=== PER-VIDEO ===")
    primary_pa_key = "aria_native_pa_mpjpe_mm" if args.dataset == "aria_val" else "pov_native_pa_mpjpe_mm"
    for seq, g in sorted(agg["per_group"].items()):
        line = f"  {seq}: PA-MPJPE={g.get(primary_pa_key, float('nan')):.2f} mm  n={g['n']}"
        if args.dataset == "aria_val":
            line += f"  (HSAM PA={g.get('hsam_pa_mpjpe_mm', float('nan')):.2f})"
        else:
            line += f"  PVE={g.get('pov_native_pve_mm', float('nan')):.2f} mm  P2D={g.get('pov_native_p2d_px', float('nan')):.1f} px"
        print(line)

    with open(args.out, "w") as f:
        json.dump({"summary": agg, "per_sample": per_sample}, f, default=float)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
