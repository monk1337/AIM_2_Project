#!/usr/bin/env python3
"""
Evaluate finetuned WiLoR on POV-Surgery test set.

Reuses the exact same evaluation pipeline as eval/run_eval_wilor.py (zero-shot),
only swapping the model weights to the finetuned checkpoint.

Usage:
    python eval_finetuned.py --ckpt-path ./output/checkpoints/best.ckpt \
                             --data-dir ../pov_surgery_data --split full
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Reuse all GT/metric utilities from our shared module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.gt_processing import (
    K, MANO_TO_OPENPOSE,
    build_gt_mano, mano_forward, transform_to_camera,
    load_gt_annotation, derive_bbox_from_2d_joints,
    procrustes_align, compute_pa_mpjpe, compute_per_finger_mpjpe,
)


def load_wilor_pipeline_with_finetuned_weights(ckpt_path, device, dtype):
    """Load wilor_mini pipeline, then swap in finetuned weights."""
    # Patch torch.load to allow pickle
    _original_load = torch.load
    def _patched_load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _original_load(*a, **kw)
    torch.load = _patched_load

    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)

    # Swap in finetuned weights
    print(f"Loading finetuned weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    missing, unexpected = pipe.wilor_model.load_state_dict(state_dict, strict=False)
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    pipe.wilor_model.eval()
    pipe.wilor_model.to(device, dtype=dtype)

    torch.load = _original_load
    return pipe


def extract_first_hand(outputs):
    """Extract predictions from first hand output (same as run_eval_wilor.py)."""
    if len(outputs) == 0:
        return None
    preds = outputs[0]["wilor_preds"]
    return {
        "joints_3d": preds["pred_keypoints_3d"][0].copy(),
        "vertices": preds["pred_vertices"][0].copy(),
        "pred_keypoints_2d": preds.get("pred_keypoints_2d"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate finetuned WiLoR on POV-Surgery")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Finetuned checkpoint path")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to pov_surgery_data/")
    parser.add_argument("--split", type=str, default="full", choices=["demo", "full"])
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_root = data_dir / "demo_data" / "POV_Surgery_data"
    mano_model_dir = data_dir / "data" / "bodymodel"

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = Path(args.ckpt_path).parent.parent / "eval_results" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    # Load model (same wilor_mini pipeline, swapped weights)
    pipe = load_wilor_pipeline_with_finetuned_weights(args.ckpt_path, device, dtype)

    # Build GT MANO
    print("Loading MANO model for GT...")
    gt_mano, tip_ids = build_gt_mano(str(mano_model_dir), device)

    # ── Collect frames (same logic as run_eval_wilor.py) ──
    if args.split == "demo":
        sequences = ["s_scalpel_3"]
        test_info = None
    else:
        test_pkl = data_root / "handoccnet_train" / "2d_repro_ho3d_style_test_cleaned.pkl"
        with open(test_pkl, "rb") as f:
            test_info = pickle.load(f)
        sequences = sorted(set(k.split("/")[0] for k in test_info.keys()))

    annotation_files = []
    if test_info is not None:
        for pkl_key, val in test_info.items():
            seq, frame_id = pkl_key.split("/")
            pkl_path = data_root / "annotation" / seq / f"{frame_id}.pkl"
            img_path = data_root / "color" / seq / f"{frame_id}.jpg"
            if pkl_path.exists() and img_path.exists():
                juv_raw = val["joints_uv"]
                juv = np.zeros_like(juv_raw, dtype=np.float32)
                juv[:, 0] = juv_raw[:, 1]
                juv[:, 1] = juv_raw[:, 0]
                juv = juv[MANO_TO_OPENPOSE]
                annotation_files.append((pkl_key, pkl_path, img_path, juv))
    else:
        for seq in sequences:
            ann_dir = data_root / "annotation" / seq
            img_dir = data_root / "color" / seq
            if not ann_dir.exists():
                continue
            for pkl_path in sorted(ann_dir.glob("*.pkl")):
                frame_id = pkl_path.stem
                img_path = img_dir / f"{frame_id}.jpg"
                if img_path.exists():
                    annotation_files.append((f"{seq}/{frame_id}", pkl_path, img_path, None))

    if args.max_frames > 0:
        annotation_files = annotation_files[:args.max_frames]

    print(f"Evaluating {len(annotation_files)} frames across {len(sequences)} sequence(s)")

    # ── Precompute GT ──
    gt_cache_path = output_dir / f"gt_cache_{args.split}.pkl"
    if gt_cache_path.exists():
        print(f"Loading cached GT from {gt_cache_path}...")
        with open(gt_cache_path, "rb") as f:
            gt_cache = pickle.load(f)
    else:
        print("Precomputing GT...")
        gt_cache = {}
        for key, pkl_path, img_path, joints_uv in annotation_files:
            gt = load_gt_annotation(pkl_path)
            gt_joints_local, gt_verts_local = mano_forward(
                gt_mano, tip_ids, gt["global_orient"], gt["hand_pose"], gt["betas"], device
            )
            gt_joints, gt_verts = transform_to_camera(gt_joints_local, gt_verts_local, gt)

            if joints_uv is not None:
                bbox = derive_bbox_from_2d_joints(joints_uv, pad_factor=1.5)
            else:
                from utils.gt_processing import project_to_2d
                joints_2d = project_to_2d(gt_joints)
                bbox = derive_bbox_from_2d_joints(joints_2d, pad_factor=1.5)

            gt_cache[key] = {
                "joints": gt_joints, "verts": gt_verts,
                "bbox": bbox, "joints_2d": joints_uv,
            }
        with open(gt_cache_path, "wb") as f:
            pickle.dump(gt_cache, f)
        print(f"  Saved GT cache: {len(gt_cache)} frames")

    bbox_failures = sum(1 for v in gt_cache.values() if v["bbox"] is None)

    # ── Evaluation loop (same as run_eval_wilor.py) ──
    all_metrics = []
    detection_failures = 0
    t_start = time.time()

    for i, (key, pkl_path, img_path, _juv) in enumerate(annotation_files):
        if (i + 1) % 200 == 0 or i == 0:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(annotation_files)}]  {fps:.1f} fps  "
                  f"det_fail={detection_failures}  bbox_fail={bbox_failures}")

        cached = gt_cache[key]
        gt_joints = cached["joints"]
        gt_verts = cached["verts"]
        bbox = cached["bbox"]

        if bbox is None:
            all_metrics.append({"frame_id": key, "detected": False, "reason": "bbox_fail"})
            continue

        image = np.array(Image.open(img_path).convert("RGB"))

        # Use pipe.predict_with_bboxes — identical to zero-shot eval
        bboxes = np.array([bbox])
        is_rights = [1.0]
        outputs = pipe.predict_with_bboxes(image, bboxes, is_rights, rescale_factor=2.5)
        pred = extract_first_hand(outputs)

        if pred is None:
            detection_failures += 1
            all_metrics.append({"frame_id": key, "detected": False, "reason": "no_detection"})
            continue

        pred_joints = pred["joints_3d"]
        pred_verts = pred["vertices"]

        # Root-relative MPJPE
        pred_root_rel = pred_joints - pred_joints[0:1]
        gt_root_rel = gt_joints - gt_joints[0:1]
        mpjpe = np.sqrt(((pred_root_rel - gt_root_rel) ** 2).sum(axis=-1)).mean() * 1000

        # PA-MPJPE
        pa_mpjpe = compute_pa_mpjpe(pred_joints, gt_joints)

        # Root-relative PVE
        pred_verts_rel = pred_verts - pred_joints[0:1]
        gt_verts_rel = gt_verts - gt_joints[0:1]
        pve = np.sqrt(((pred_verts_rel - gt_verts_rel) ** 2).sum(axis=-1)).mean() * 1000

        # PA-PVE
        pred_verts_aligned = procrustes_align(pred_verts, gt_verts)
        pa_pve = np.sqrt(((pred_verts_aligned - gt_verts) ** 2).sum(axis=-1)).mean() * 1000

        # Per-finger
        per_finger = compute_per_finger_mpjpe(pred_joints, gt_joints)

        # P2D (using wilor_mini's proper projection)
        gt_2d_stored = cached.get("joints_2d")
        wilor_kp2d = pred.get("pred_keypoints_2d")
        if gt_2d_stored is not None and wilor_kp2d is not None:
            pred_2d = wilor_kp2d[0]
            p2d_error = np.sqrt(((pred_2d - gt_2d_stored) ** 2).sum(axis=-1)).mean()
        else:
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

    # ── Aggregate results ──
    elapsed = time.time() - t_start
    detected = [m for m in all_metrics if m.get("detected", False)]
    total = len(all_metrics)
    n_det = len(detected)

    print("\n" + "=" * 60)
    print(f"RESULTS: WiLoR (finetuned on POV-Surgery)")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Split: {args.split} ({len(sequences)} sequences)")
    print("=" * 60)
    print(f"Total frames:       {total}")
    print(f"Detected:           {n_det} ({100*n_det/total:.1f}%)")
    print(f"Detection failures: {detection_failures}")
    print(f"Bbox failures:      {bbox_failures}")
    print(f"Time:               {elapsed:.1f}s ({total/elapsed:.1f} fps)")

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
            ("P2d (px)", p2d_vals),
        ]:
            if vals:
                arr = np.array(vals)
                print(f"{name:<20} {arr.mean():>8.2f} {np.median(arr):>8.2f} {arr.std():>8.2f}")

        # Per-finger
        print(f"\n{'Finger':<12} {'MPJPE (mm)':>12}")
        print("-" * 26)
        for fn in ["thumb", "index", "middle", "ring", "pinky"]:
            vals = [m["per_finger"][fn] for m in detected]
            print(f"{fn:<12} {np.mean(vals):>12.2f}")

        print(f"\nTarget (CPCI Table 2): MPJPE=13.72  PA-MPJPE=4.33  PVE=12.91  PA-PVE=4.20  P2d=18.48")
        print(f"Zero-shot baseline:    MPJPE=50.36  PA-MPJPE=10.69 PVE=47.92  PA-PVE=10.01 P2d=26.00")

    # Save results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": "WiLoR (finetuned)",
            "checkpoint": str(args.ckpt_path),
            "split": args.split,
            "total_frames": total,
            "detected": n_det,
            "detection_failures": detection_failures,
            "time_seconds": elapsed,
            "metrics_summary": {
                "mpjpe_mean": float(np.mean(mpjpe_vals)) if n_det else None,
                "pa_mpjpe_mean": float(np.mean(pa_mpjpe_vals)) if n_det else None,
                "pve_mean": float(np.mean(pve_vals)) if n_det else None,
                "pa_pve_mean": float(np.mean(pa_pve_vals)) if n_det else None,
                "p2d_mean": float(np.mean(p2d_vals)) if p2d_vals else None,
            },
            "per_frame": all_metrics,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
