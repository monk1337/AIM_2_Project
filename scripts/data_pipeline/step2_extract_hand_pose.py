"""
Step 2: Extract 3D hand keypoints + MANO parameters from filtered surgical frames.

Uses WiLoR to extract per-hand:
  - 21 3D keypoints
  - MANO pose (theta, 48 dims) + shape (beta, 10 dims)
  - Camera parameters
  - 2D keypoints for visualization

Two strategies for getting surgeon's own hands:
  A) detect-then-filter (default): WiLoR YOLO detects all hands, then IoU-match
     against EgoSurgery "Own hands" annotations to keep only surgeon's hands.
  B) crop-regress (--crop-regress): Skip YOLO entirely. Use EgoSurgery GT bboxes
     to directly feed each hand crop to WiLoR's MANO regressor. Often more accurate
     because the model sees a cleaner, GT-centered crop.

Usage:
  python scripts/step2_extract_hand_pose.py --visualize
  python scripts/step2_extract_hand_pose.py --crop-regress --visualize  # GT bbox mode
  python scripts/step2_extract_hand_pose.py --no-filter   # keep all detected hands
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Patch torch.load for compatibility with newer PyTorch (>=2.6)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_filtered_frames(json_path: str) -> list[dict]:
    """Load Step 1 output."""
    with open(json_path) as f:
        return json.load(f)


def load_hand_bboxes_for_frame(video_id: str, frame_id: str) -> list[dict]:
    """Load hand bounding boxes for a specific frame from EgoSurgery annotations."""
    bbox_file = (
        PROJECT_ROOT / "full_data" / "annotations" / "bbox" / "by_video"
        / "hands" / video_id / "annotations.json"
    )
    if not bbox_file.exists():
        return []

    with open(bbox_file) as f:
        data = json.load(f)

    # Map image filenames to ids
    filename_to_id = {}
    for img in data["images"]:
        name = img["file_name"].replace(".jpg", "")
        filename_to_id[name] = img["id"]

    # Category mapping
    cat_names = {c["id"]: c["name"] for c in data["categories"]}

    image_id = filename_to_id.get(frame_id)
    if image_id is None:
        return []

    bboxes = []
    for ann in data["annotations"]:
        if ann["image_id"] == image_id:
            cat_name = cat_names.get(ann["category_id"], "unknown")
            # Only keep surgeon's own hands
            if "Own hands" in cat_name:
                is_right = "right" in cat_name.lower()
                x, y, w, h = ann["bbox"]
                bboxes.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "is_right": is_right,
                    "category": cat_name,
                })
    return bboxes


def load_video_annotations(video_id: str) -> dict:
    """Load and cache full annotation data for a video. Returns raw COCO dict."""
    bbox_file = (
        PROJECT_ROOT / "full_data" / "annotations" / "bbox" / "by_video"
        / "hands" / video_id / "annotations.json"
    )
    if not bbox_file.exists():
        return {}
    with open(bbox_file) as f:
        return json.load(f)


def get_own_hand_bboxes(ann_data: dict, frame_id: str) -> list[dict]:
    """Get 'Own hands' bboxes for a frame from pre-loaded annotation data."""
    if not ann_data:
        return []

    # Map filenames to image ids
    filename_to_id = {}
    for img in ann_data["images"]:
        name = img["file_name"].replace(".jpg", "")
        filename_to_id[name] = img["id"]

    cat_names = {c["id"]: c["name"] for c in ann_data["categories"]}

    image_id = filename_to_id.get(frame_id)
    if image_id is None:
        return []

    bboxes = []
    for ann in ann_data["annotations"]:
        if ann["image_id"] == image_id:
            cat_name = cat_names.get(ann["category_id"], "unknown")
            if "Own hands" in cat_name:
                is_right = "right" in cat_name.lower()
                x, y, w, h = ann["bbox"]
                # Convert COCO [x, y, w, h] to [x1, y1, x2, y2]
                bboxes.append({
                    "bbox_xyxy": [x, y, x + w, y + h],
                    "is_right": is_right,
                    "category": cat_name,
                })
    return bboxes


def compute_iou(box_a: list, box_b: list) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def filter_to_own_hands(
    wilor_outputs: list,
    ann_data: dict,
    frame_id: str,
    iou_thresh: float = 0.3,
) -> list:
    """Filter WiLoR detections to only surgeon's own hands using EgoSurgery annotations.

    Matches each WiLoR detection bbox against 'Own hands' annotation bboxes via IoU.
    Also updates is_right from the annotation (more reliable than YOLO classifier).
    """
    own_bboxes = get_own_hand_bboxes(ann_data, frame_id)
    if not own_bboxes:
        return wilor_outputs  # No annotations available, keep all

    filtered = []
    matched_ann_indices = set()

    for det in wilor_outputs:
        det_bbox = det["hand_bbox"]  # [x1, y1, x2, y2] from YOLO

        best_iou = 0.0
        best_ann_idx = -1
        for ann_idx, ann in enumerate(own_bboxes):
            iou = compute_iou(det_bbox, ann["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_ann_idx = ann_idx

        if best_iou >= iou_thresh and best_ann_idx not in matched_ann_indices:
            # Update is_right from annotation (more reliable)
            det["is_right"] = own_bboxes[best_ann_idx]["is_right"]
            filtered.append(det)
            matched_ann_indices.add(best_ann_idx)

    return filtered


def to_list(val):
    """Convert tensor or numpy array to Python list."""
    if hasattr(val, "cpu"):
        return val.cpu().numpy().tolist()
    elif isinstance(val, np.ndarray):
        return val.tolist()
    return val


def extract_mano_from_outputs(outputs: list) -> list[dict]:
    """Extract MANO parameters and keypoints from WiLoR output.

    WiLoR output structure per hand:
      wilor_preds:
        global_orient: (1, 1, 3), wrist rotation axis-angle
        hand_pose: (1, 15, 3), 15 joint rotations axis-angle
        betas: (1, 10), shape coefficients
        pred_cam_t_full: (1, 3), camera-space translation
        pred_keypoints_3d: (1, 21, 3), 3D keypoints
        pred_keypoints_2d: (1, 21, 2), 2D keypoints
        pred_vertices: (1, 778, 3), mesh vertices
        scaled_focal_length: float
    """
    results = []
    for out in outputs:
        preds = out["wilor_preds"]

        # Flatten hand_pose from (1, 15, 3) -> 45 dims
        hand_pose_flat = np.array(preds["hand_pose"][0]).reshape(-1).tolist()  # 45 dims
        global_orient_flat = np.array(preds["global_orient"][0]).reshape(-1).tolist()  # 3 dims

        result = {
            "is_right": bool(out["is_right"]),
            # MANO parameters
            "mano_pose": hand_pose_flat,  # 45 dims (15 joints * 3 axis-angle)
            "mano_shape": to_list(preds["betas"][0]),  # 10 dims
            "mano_global_orient": global_orient_flat,  # 3 dims
            # 3D keypoints (camera space)
            "keypoints_3d": to_list(preds["pred_keypoints_3d"][0]),  # 21 x 3
            # 2D keypoints (image space)
            "keypoints_2d": to_list(preds["pred_keypoints_2d"][0]),  # 21 x 2
            # Camera translation
            "cam_t_full": to_list(preds["pred_cam_t_full"][0]),  # 3 dims
            # Vertices (for visualization, optional)
            "pred_vertices": to_list(preds["pred_vertices"][0]),  # 778 x 3
            # Focal length used
            "scaled_focal_length": float(preds["scaled_focal_length"]),
        }
        results.append(result)
    return results


def convert_to_vitra_format(hand_result: dict) -> dict:
    """
    Convert WiLoR MANO output to VITRA's expected 61-dim format.
    VITRA expects: translation(3) + rotation(3) + joint_angles(45) + shape(10) = 61
    """
    cam_t = hand_result["cam_t_full"]  # 3 dims - translation
    global_orient = hand_result["mano_global_orient"]  # 3 dims - rotation (axis-angle)
    hand_pose = hand_result["mano_pose"]  # 45 dims - joint angles (15 joints x 3)
    betas = hand_result["mano_shape"]  # 10 dims - shape

    # VITRA format: [translation(3), rotation(3), joint_angles(45), shape(10)]
    vitra_params = cam_t + global_orient + hand_pose + betas  # 3+3+45+10 = 61

    return {
        "vitra_params": vitra_params,  # 61 dims
        "translation": cam_t,
        "rotation": global_orient,
        "joint_angles": hand_pose,
        "shape": betas,
        "is_right": hand_result["is_right"],
    }


def draw_keypoints(image: np.ndarray, keypoints_2d: list, is_right: bool) -> np.ndarray:
    """Draw 21 hand keypoints on image."""
    img = image.copy()

    # Color: blue for left, red for right
    color = (0, 0, 255) if is_right else (255, 0, 0)

    # MANO joint connections
    connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]

    # Draw connections
    for i, j in connections:
        pt1 = (int(keypoints_2d[i][0]), int(keypoints_2d[i][1]))
        pt2 = (int(keypoints_2d[j][0]), int(keypoints_2d[j][1]))
        cv2.line(img, pt1, pt2, color, 2)

    # Draw keypoints
    for idx, kp in enumerate(keypoints_2d):
        x, y = int(kp[0]), int(kp[1])
        # Fingertips are brighter
        if idx in [4, 8, 12, 16, 20]:
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        else:
            cv2.circle(img, (x, y), 3, color, -1)

    return img


def process_frames(
    filtered_frames: list[dict],
    pipe,
    crop_regress: bool = False,
    filter_own_hands: bool = True,
    iou_thresh: float = 0.3,
    visualize: bool = False,
    output_dir: Path = None,
):
    """Process all filtered frames through WiLoR."""
    all_results = []
    vis_dir = output_dir / "visualizations" if visualize else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Cache annotation data per video to avoid re-loading
    ann_cache = {}

    for idx, frame_info in enumerate(filtered_frames):
        frame_id = frame_info["frame_id"]
        video_id = frame_info["video_id"]
        image_path = PROJECT_ROOT / frame_info["image_path"]

        print(f"\n[{idx+1}/{len(filtered_frames)}] Processing {frame_id} (phase: {frame_info['phase']})")

        if not image_path.exists():
            print(f"  WARNING: Image not found: {image_path}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  WARNING: Could not read image: {image_path}")
            continue

        # Load annotation data for this video (cached)
        if (crop_regress or filter_own_hands) and video_id not in ann_cache:
            ann_cache[video_id] = load_video_annotations(video_id)

        t0 = time.time()

        if crop_regress:
            # Mode B: Use GT bboxes directly, skip YOLO
            own_bboxes = get_own_hand_bboxes(ann_cache.get(video_id, {}), frame_id)
            if not own_bboxes:
                print(f"  WARNING: No GT bboxes for {frame_id}")
                outputs = []
            else:
                bboxes_xyxy = np.array([b["bbox_xyxy"] for b in own_bboxes])
                is_rights = [float(b["is_right"]) for b in own_bboxes]
                outputs = pipe.predict_with_bboxes(image, bboxes_xyxy, is_rights)
                elapsed = time.time() - t0
                print(f"  WiLoR crop-regress: {elapsed:.2f}s, {len(outputs)} own hand(s) from GT bboxes")
        else:
            # Mode A: YOLO detect then filter
            outputs = pipe.predict(image)
            elapsed = time.time() - t0
            n_raw = len(outputs)

            if filter_own_hands and outputs:
                outputs = filter_to_own_hands(
                    outputs, ann_cache.get(video_id, {}), frame_id, iou_thresh
                )
                print(f"  WiLoR detect-filter: {elapsed:.2f}s, detected {n_raw}, kept {len(outputs)} own hand(s)")
            else:
                print(f"  WiLoR inference: {elapsed:.2f}s, detected {n_raw} hand(s)")

        if not outputs:
            print(f"  WARNING: No hands detected in {frame_id}")
            all_results.append({
                "frame_id": frame_id,
                "video_id": video_id,
                "phase": frame_info["phase"],
                "image_path": str(frame_info["image_path"]),
                "hands_detected": 0,
                "hands": [],
                "vitra_params": {},
            })
            continue

        # Extract MANO parameters
        hands = extract_mano_from_outputs(outputs)

        # Convert to VITRA format
        vitra_params = {}
        for hand in hands:
            side = "right" if hand["is_right"] else "left"
            vitra_data = convert_to_vitra_format(hand)
            vitra_params[side] = vitra_data
            print(f"  {side} hand: MANO params extracted (61 dims)")

        # Store results (exclude large vertex data for JSON size)
        hands_compact = []
        for h in hands:
            h_compact = {k: v for k, v in h.items() if k != "pred_vertices"}
            hands_compact.append(h_compact)

        result = {
            "frame_id": frame_id,
            "video_id": video_id,
            "phase": frame_info["phase"],
            "image_path": str(frame_info["image_path"]),
            "hands_detected": len(outputs),
            "hands": hands_compact,
            "vitra_params": {
                side: {
                    "params_61": data["vitra_params"],
                    "translation": data["translation"],
                    "rotation": data["rotation"],
                    "joint_angles": data["joint_angles"],
                    "shape": data["shape"],
                    "is_right": data["is_right"],
                }
                for side, data in vitra_params.items()
            },
        }
        all_results.append(result)

        # Visualization
        if visualize:
            vis_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for hand in hands:
                vis_image = draw_keypoints(
                    vis_image, hand["keypoints_2d"], hand["is_right"]
                )
            vis_path = vis_dir / f"{frame_id}_keypoints.jpg"
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"  Saved visualization: {vis_path.name}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Step 2: Extract hand pose with WiLoR")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "pipeline_output" / "step1_filtered" / "selected_frames.json"),
        help="Path to Step 1 output JSON",
    )
    parser.add_argument("--crop-regress", action="store_true",
                        help="Use GT bboxes directly (skip YOLO detection, feed crops to WiLoR regressor)")
    parser.add_argument("--no-filter", action="store_true", help="Skip own-hand filtering (keep all detected hands)")
    parser.add_argument("--iou-thresh", type=float, default=0.3, help="IoU threshold for own-hand matching (default: 0.3)")
    parser.add_argument("--visualize", action="store_true", help="Save visualization images with keypoints")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device")
    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Use float32 for CPU/MPS, float16 for CUDA
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Device: {device}, dtype: {dtype}")

    # Load filtered frames from Step 1
    filtered_frames = load_filtered_frames(args.input)
    print(f"Loaded {len(filtered_frames)} filtered frames from Step 1")

    # Initialize WiLoR pipeline
    print("Loading WiLoR model (will download on first run)...")
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    print("WiLoR model loaded!")

    # Output directory
    output_dir = PROJECT_ROOT / "pipeline_output" / "step2_hand_pose"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process frames
    filter_own = not args.no_filter
    if args.crop_regress:
        print("Mode: CROP-REGRESS (GT bboxes → WiLoR regressor, no YOLO)")
    elif filter_own:
        print(f"Mode: DETECT-FILTER (YOLO → IoU filter, threshold: {args.iou_thresh})")
    else:
        print("Mode: DETECT-ALL (no filtering)")

    results = process_frames(
        filtered_frames,
        pipe,
        crop_regress=args.crop_regress,
        filter_own_hands=filter_own,
        iou_thresh=args.iou_thresh,
        visualize=args.visualize,
        output_dir=output_dir,
    )

    # Save results
    output_path = output_dir / "hand_pose_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {output_path}")

    # Summary
    total_hands = sum(r["hands_detected"] for r in results)
    frames_with_hands = sum(1 for r in results if r["hands_detected"] > 0)
    bimanual = sum(1 for r in results if r["hands_detected"] >= 2)
    vitra_ready = sum(1 for r in results if r["vitra_params"])
    has_left = sum(1 for r in results if "left" in r["vitra_params"])
    has_right = sum(1 for r in results if "right" in r["vitra_params"])

    print(f"\n{'='*60}")
    print(f"STEP 2 SUMMARY")
    print(f"{'='*60}")
    print(f"  Frames processed: {len(results)}")
    print(f"  Frames with hands detected: {frames_with_hands}/{len(results)}")
    print(f"  Total own hands detected: {total_hands}")
    print(f"  Bimanual frames: {bimanual}")
    print(f"  Left hand frames: {has_left}")
    print(f"  Right hand frames: {has_right}")
    print(f"  VITRA-ready frames: {vitra_ready}")
    if filter_own:
        print(f"  Own-hand filtering: ENABLED (IoU >= {args.iou_thresh})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
