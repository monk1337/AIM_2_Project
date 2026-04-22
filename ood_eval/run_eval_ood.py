#!/usr/bin/env python3
"""Evaluate WiLoR on OOD datasets in the common format.

Consumes common_format/<dataset>/{images/, samples.pkl}. Always uses GT-bbox
crop-regress mode. Supports both right and left hands via horizontal flipping.

Usage:
    python run_eval_ood.py \\
        --dataset-dir common_format/aria \\
        --output-dir  results/aria_raw \\
        [--ckpt-path /path/to/finetuned.ckpt] \\
        [--max-samples 0] [--device cuda]
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


# ── Metrics ────────────────────────────────────────────────────────────
def procrustes_align(S1, S2):
    """Align S1 to S2 via similarity transform (rotation + scale + translation)."""
    mu1 = S1.mean(axis=0, keepdims=True)
    mu2 = S2.mean(axis=0, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = np.sum(X1 ** 2)
    if var1 < 1e-10:
        return S1

    K_mat = X1.T @ X2
    U, s, Vt = np.linalg.svd(K_mat)
    Z = np.eye(3)
    Z[-1, -1] *= np.sign(np.linalg.det(U @ Vt))
    R = Vt.T @ Z @ U.T
    scale = np.trace(R @ K_mat) / var1
    t = mu2 - scale * (R @ mu1.T).T

    S1_hat = scale * (S1 @ R.T) + t
    return S1_hat


def compute_mpjpe(pred, gt):
    return np.sqrt(((pred - gt) ** 2).sum(axis=-1)).mean() * 1000


def compute_pa_mpjpe(pred, gt):
    pred_aligned = procrustes_align(pred, gt)
    return np.sqrt(((pred_aligned - gt) ** 2).sum(axis=-1)).mean() * 1000


def compute_per_finger_mpjpe(pred, gt):
    """Per-finger PA-MPJPE (PA-aligned then per-finger error, in mm)."""
    pred_aligned = procrustes_align(pred, gt)
    fingers = {
        "thumb": slice(1, 5),
        "index": slice(5, 9),
        "middle": slice(9, 13),
        "ring": slice(13, 17),
        "pinky": slice(17, 21),
    }
    result = {}
    for name, sl in fingers.items():
        err = np.sqrt(((pred_aligned[sl] - gt[sl]) ** 2).sum(axis=-1)).mean() * 1000
        result[name] = err
    return result


# ── WiLoR Prediction ──────────────────────────────────────────────────
def extract_first_hand(outputs):
    if len(outputs) == 0:
        return None
    preds = outputs[0]["wilor_preds"]
    kp2d = preds.get("pred_keypoints_2d")
    return {
        "joints_3d": preds["pred_keypoints_3d"][0].copy(),
        "vertices": preds["pred_vertices"][0].copy(),
        "global_orient": preds["global_orient"][0].copy(),
        "hand_pose": preds["hand_pose"][0].copy(),
        "betas": preds["betas"][0].copy(),
        "joints_2d": kp2d[0].copy() if kp2d is not None else None,
    }


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate WiLoR on OOD common-format datasets")
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Path to common_format/<dataset> dir containing samples.pkl and images/")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results JSON")
    parser.add_argument("--ckpt-path", type=str, default=None,
                        help="Path to finetuned checkpoint. If omitted, uses off-the-shelf weights.")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples (0 = all)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu/mps)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    samples_pkl = dataset_dir / "samples.pkl"
    if not samples_pkl.exists():
        print(f"ERROR: samples.pkl not found at {samples_pkl}")
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    with open(samples_pkl, "rb") as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} samples from {samples_pkl}")

    if args.max_samples > 0:
        samples = samples[: args.max_samples]
        print(f"Processing first {len(samples)} samples")

    # Init WiLoR pipeline
    print("Loading WiLoR pipeline...")
    _original_load = torch.load
    def _patched_load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _original_load(*a, **kw)
    torch.load = _patched_load

    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
    torch.load = _original_load

    if args.ckpt_path:
        print(f"Loading finetuned weights from {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)
        missing, unexpected = pipe.wilor_model.load_state_dict(state_dict, strict=False)
        print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        pipe.wilor_model.eval()
        pipe.wilor_model.to(device, dtype=dtype)

    model_label = "WiLoR (finetuned)" if args.ckpt_path else "WiLoR (off-the-shelf)"
    print(f"{model_label} loaded.")

    # ── Evaluation loop ────────────────────────────────────────────────
    all_metrics = []
    detection_failures = 0
    t_start = time.time()

    for i, sample in enumerate(samples):
        if (i + 1) % 200 == 0 or i == 0:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(samples)}]  {fps:.1f} samples/s  det_fail={detection_failures}")

        frame_id = sample["frame_id"]
        rel_img = sample["image_path"]
        img_path = (samples_pkl.parent / rel_img).resolve()
        bbox = np.array(sample["bbox"], dtype=np.float32).copy()
        is_right = bool(sample["is_right"])
        hand_side = sample.get("hand_side", "right" if is_right else "left")
        gt_joints_3d = sample.get("joints_3d")
        gt_vertices_3d = sample.get("vertices_3d")
        gt_joints_2d = sample.get("joints_2d")
        frame_trust = bool(sample.get("joints_3d_frame_trustworthy", False))

        if not img_path.exists():
            print(f"  WARNING: image not found: {img_path}")
            all_metrics.append({
                "frame_id": frame_id, "image_path": str(rel_img),
                "hand_side": hand_side, "is_right": is_right,
                "detected": False, "reason": "missing_image",
                "mpjpe": None, "pa_mpjpe": None, "pve": None,
                "pa_pve": None, "p2d": None, "per_finger": None,
                "pred_joints_2d": None, "pred_joints_3d": None,
                "bbox": bbox.tolist(),
                "gt_joints_2d": gt_joints_2d.tolist() if gt_joints_2d is not None else None,
                "gt_joints_3d": gt_joints_3d.tolist() if gt_joints_3d is not None else None,
            })
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        H, W = image.shape[:2]

        # For left hands, flip the image + bbox horizontally and run as right hand.
        # Un-flip predictions afterwards. GT stays in original orientation.
        if is_right:
            image_in = image
            bbox_in = bbox
        else:
            image_in = image[:, ::-1, :].copy()
            x1, y1, x2, y2 = bbox
            bbox_in = np.array([W - x2, y1, W - x1, y2], dtype=np.float32)

        bboxes = np.array([bbox_in])
        is_rights = [1.0]
        try:
            outputs = pipe.predict_with_bboxes(image_in, bboxes, is_rights, rescale_factor=2.5)
        except Exception as e:
            print(f"  ERROR predicting on {frame_id}: {e}")
            outputs = []
        pred = extract_first_hand(outputs)

        if pred is None:
            detection_failures += 1
            all_metrics.append({
                "frame_id": frame_id, "image_path": str(rel_img),
                "hand_side": hand_side, "is_right": is_right,
                "detected": False, "reason": "no_prediction",
                "mpjpe": None, "pa_mpjpe": None, "pve": None,
                "pa_pve": None, "p2d": None, "per_finger": None,
                "pred_joints_2d": None, "pred_joints_3d": None,
                "bbox": bbox.tolist(),
                "gt_joints_2d": gt_joints_2d.tolist() if gt_joints_2d is not None else None,
                "gt_joints_3d": gt_joints_3d.tolist() if gt_joints_3d is not None else None,
            })
            continue

        pred_joints = pred["joints_3d"].astype(np.float32)
        pred_verts = pred["vertices"].astype(np.float32)
        pred_2d = pred["joints_2d"].astype(np.float32) if pred["joints_2d"] is not None else None

        # Un-flip predictions for left hand (so everything lives in original image/world coords)
        if not is_right:
            if pred_2d is not None:
                pred_2d[:, 0] = (W - 1) - pred_2d[:, 0]
            pred_joints[:, 0] *= -1
            pred_verts[:, 0] *= -1

        # Metrics
        mpjpe = None
        pa_mpjpe = None
        pve = None
        pa_pve = None
        p2d_error = None
        per_finger = None

        if gt_joints_3d is not None:
            gt_j = np.asarray(gt_joints_3d, dtype=np.float32)
            pa_mpjpe = float(compute_pa_mpjpe(pred_joints, gt_j))
            per_finger = {k: float(v) for k, v in compute_per_finger_mpjpe(pred_joints, gt_j).items()}
            if frame_trust:
                pred_rr = pred_joints - pred_joints[0:1]
                gt_rr = gt_j - gt_j[0:1]
                mpjpe = float(np.sqrt(((pred_rr - gt_rr) ** 2).sum(axis=-1)).mean() * 1000)

        if gt_vertices_3d is not None:
            gt_v = np.asarray(gt_vertices_3d, dtype=np.float32)
            pred_verts_aligned = procrustes_align(pred_verts, gt_v)
            pa_pve = float(np.sqrt(((pred_verts_aligned - gt_v) ** 2).sum(axis=-1)).mean() * 1000)
            if frame_trust and gt_joints_3d is not None:
                gt_j = np.asarray(gt_joints_3d, dtype=np.float32)
                pred_verts_rel = pred_verts - pred_joints[0:1]
                gt_verts_rel = gt_v - gt_j[0:1]
                pve = float(np.sqrt(((pred_verts_rel - gt_verts_rel) ** 2).sum(axis=-1)).mean() * 1000)

        if gt_joints_2d is not None and pred_2d is not None:
            gt2 = np.asarray(gt_joints_2d, dtype=np.float32)
            p2d_error = float(np.sqrt(((pred_2d - gt2) ** 2).sum(axis=-1)).mean())

        all_metrics.append({
            "frame_id": frame_id,
            "image_path": str(rel_img),
            "hand_side": hand_side,
            "is_right": is_right,
            "detected": True,
            "mpjpe": mpjpe,
            "pa_mpjpe": pa_mpjpe,
            "pve": pve,
            "pa_pve": pa_pve,
            "p2d": p2d_error,
            "per_finger": per_finger,
            "pred_joints_2d": pred_2d.tolist() if pred_2d is not None else None,
            "pred_joints_3d": pred_joints.tolist(),
            "bbox": bbox.tolist(),
            "gt_joints_2d": gt_joints_2d.tolist() if gt_joints_2d is not None else None,
            "gt_joints_3d": gt_joints_3d.tolist() if gt_joints_3d is not None else None,
        })

    # ── Aggregate ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    total = len(all_metrics)

    def _collect(key):
        return [m[key] for m in all_metrics if m.get(key) is not None]

    mpjpe_vals = _collect("mpjpe")
    pa_mpjpe_vals = _collect("pa_mpjpe")
    pve_vals = _collect("pve")
    pa_pve_vals = _collect("pa_pve")
    p2d_vals = _collect("p2d")

    evaluated = sum(
        1 for m in all_metrics
        if any(m.get(k) is not None for k in ("mpjpe", "pa_mpjpe", "pve", "pa_pve", "p2d"))
    )

    dataset_name = dataset_dir.name
    print("\n" + "=" * 60)
    print(f"RESULTS: {model_label} (crop-regress)")
    print(f"Dataset: {dataset_name}")
    print("=" * 60)
    print(f"Total samples:      {total}")
    print(f"Evaluated:          {evaluated}")
    print(f"Detection failures: {detection_failures}")
    print(f"Time:               {elapsed:.1f}s ({total/elapsed:.1f} samples/s)")

    print(f"\n{'Metric':<20} {'Mean':>8} {'Median':>8} {'Std':>8} {'N':>6}")
    print("-" * 54)
    for name, vals in [
        ("MPJPE (mm)", mpjpe_vals),
        ("PA-MPJPE (mm)", pa_mpjpe_vals),
        ("PVE (mm)", pve_vals),
        ("PA-PVE (mm)", pa_pve_vals),
        ("P2d (px)", p2d_vals),
    ]:
        if vals:
            arr = np.array(vals)
            print(f"{name:<20} {arr.mean():>8.2f} {np.median(arr):>8.2f} {arr.std():>8.2f} {len(arr):>6d}")
        else:
            print(f"{name:<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {0:>6d}")

    # Per-finger aggregate (PA-MPJPE)
    per_finger_samples = [m["per_finger"] for m in all_metrics if m.get("per_finger") is not None]
    if per_finger_samples:
        print(f"\n{'Finger':<12} {'PA-MPJPE (mm)':>14}")
        print("-" * 28)
        for fn in ["thumb", "index", "middle", "ring", "pinky"]:
            vals = [pf[fn] for pf in per_finger_samples]
            print(f"{fn:<12} {np.mean(vals):>14.2f}")
    else:
        print("\nPer-finger: N/A")

    results_path = output_dir / "wilor_ood_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": model_label,
            "dataset": dataset_name,
            "total_samples": total,
            "evaluated": evaluated,
            "detection_failures": detection_failures,
            "time_seconds": elapsed,
            "metrics_summary": {
                "mpjpe_mean": float(np.mean(mpjpe_vals)) if mpjpe_vals else None,
                "pa_mpjpe_mean": float(np.mean(pa_mpjpe_vals)) if pa_mpjpe_vals else None,
                "pve_mean": float(np.mean(pve_vals)) if pve_vals else None,
                "pa_pve_mean": float(np.mean(pa_pve_vals)) if pa_pve_vals else None,
                "p2d_mean": float(np.mean(p2d_vals)) if p2d_vals else None,
            },
            "per_sample": all_metrics,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
