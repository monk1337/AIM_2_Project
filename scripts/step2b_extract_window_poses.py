"""
Step 2b: Extract MANO hand poses for all frames in step1b windows.

Runs WiLoR crop-regress on every frame referenced by windows.json,
using EgoSurgery GT bboxes (skip YOLO). Output is keyed by frame_id
for O(1) lookup during episode packaging in step3.

Usage:
  .venv/bin/python scripts/step2b_extract_window_poses.py --video 01
  .venv/bin/python scripts/step2b_extract_window_poses.py --all-videos
  .venv/bin/python scripts/step2b_extract_window_poses.py --video 01 --visualize
"""

import argparse
import json
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


def to_list(val):
    """Convert tensor or numpy array to Python list."""
    if hasattr(val, "cpu"):
        return val.cpu().numpy().tolist()
    elif isinstance(val, np.ndarray):
        return val.tolist()
    return val


def load_video_annotations(video_id: str) -> dict:
    """Load full COCO annotation data for a video."""
    bbox_file = (
        PROJECT_ROOT / "full_data" / "annotations" / "bbox" / "by_video"
        / "hands" / video_id / "annotations.json"
    )
    if not bbox_file.exists():
        return {}
    with open(bbox_file) as f:
        return json.load(f)


def build_bbox_index(ann_data: dict) -> dict[str, list[dict]]:
    """Build frame_id → [(bbox_xyxy, is_right), ...] index in one pass.

    Returns dict mapping frame_id to list of own-hand bboxes.
    O(N) over all annotations once, then O(1) lookup per frame.
    """
    if not ann_data:
        return {}

    # Map image_id → frame_id (filename without .jpg)
    image_id_to_frame_id = {}
    for img in ann_data["images"]:
        name = img["file_name"].replace(".jpg", "")
        image_id_to_frame_id[img["id"]] = name

    # Category mapping
    cat_names = {c["id"]: c["name"] for c in ann_data["categories"]}

    # Single pass over annotations
    index: dict[str, list[dict]] = {}
    for ann in ann_data["annotations"]:
        cat_name = cat_names.get(ann["category_id"], "unknown")
        if "Own hands" not in cat_name:
            continue

        frame_id = image_id_to_frame_id.get(ann["image_id"])
        if frame_id is None:
            continue

        is_right = "right" in cat_name.lower()
        x, y, w, h = ann["bbox"]
        bbox_xyxy = [x, y, x + w, y + h]

        if frame_id not in index:
            index[frame_id] = []
        index[frame_id].append({
            "bbox_xyxy": bbox_xyxy,
            "is_right": is_right,
        })

    return index


def extract_frame_pose(pipe, image: np.ndarray, bboxes: list[dict]) -> dict:
    """Run WiLoR crop-regress on a single frame with given GT bboxes.

    Returns dict with 'left' and/or 'right' hand data.
    """
    if not bboxes:
        return {}

    bboxes_xyxy = np.array([b["bbox_xyxy"] for b in bboxes])
    is_rights = [float(b["is_right"]) for b in bboxes]

    outputs = pipe.predict_with_bboxes(image, bboxes_xyxy, is_rights)

    result = {}
    for out in outputs:
        preds = out["wilor_preds"]
        side = "right" if out["is_right"] else "left"

        result[side] = {
            "global_orient": to_list(np.array(preds["global_orient"][0]).reshape(-1)),  # 3
            "hand_pose": to_list(np.array(preds["hand_pose"][0]).reshape(-1)),  # 45
            "betas": to_list(preds["betas"][0]),  # 10
            "cam_t_full": to_list(preds["pred_cam_t_full"][0]),  # 3
            "keypoints_3d": to_list(preds["pred_keypoints_3d"][0]),  # 21x3
            "keypoints_2d": to_list(preds["pred_keypoints_2d"][0]),  # 21x2
        }

    return result


def draw_keypoints(image: np.ndarray, keypoints_2d: list, is_right: bool) -> np.ndarray:
    """Draw 21 hand keypoints on image."""
    img = image.copy()
    color = (0, 0, 255) if is_right else (255, 0, 0)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    for i, j in connections:
        pt1 = (int(keypoints_2d[i][0]), int(keypoints_2d[i][1]))
        pt2 = (int(keypoints_2d[j][0]), int(keypoints_2d[j][1]))
        cv2.line(img, pt1, pt2, color, 2)
    for idx, kp in enumerate(keypoints_2d):
        x, y = int(kp[0]), int(kp[1])
        if idx in [4, 8, 12, 16, 20]:
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        else:
            cv2.circle(img, (x, y), 3, color, -1)
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Step 2b: Extract MANO poses for all window frames via WiLoR crop-regress")
    parser.add_argument("--video", default="01", help="Video ID to process (default: 01)")
    parser.add_argument("--all-videos", action="store_true", help="Process all videos in windows.json")
    parser.add_argument("--windows-json",
                        default=str(PROJECT_ROOT / "pipeline_output" / "step1b_windows" / "windows.json"),
                        help="Path to windows.json from step1b")
    parser.add_argument("--visualize", action="store_true", help="Save spot-check visualizations")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    # Load windows
    with open(args.windows_json) as f:
        windows_data = json.load(f)
    all_windows = windows_data["windows"]

    # Filter to requested video(s)
    if args.all_videos:
        video_ids = sorted(set(w["video_id"] for w in all_windows))
    else:
        video_ids = [args.video]

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
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    # Initialize WiLoR once
    print("Loading WiLoR model...")
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    print("WiLoR model loaded!")

    output_dir = PROJECT_ROOT / "pipeline_output" / "step2b_window_poses"
    output_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = output_dir / "visualizations" if args.visualize else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for video_id in video_ids:
        print(f"\n{'=' * 60}")
        print(f"Processing video {video_id}")
        print(f"{'=' * 60}")

        # Collect unique frame_ids for this video
        video_windows = [w for w in all_windows if w["video_id"] == video_id]
        if not video_windows:
            print(f"  No windows for video {video_id}, skipping")
            continue

        unique_frames: dict[str, str] = {}  # frame_id → image_path
        for w in video_windows:
            for fid, ipath in zip(w["frame_ids"], w["image_paths"]):
                if fid not in unique_frames:
                    unique_frames[fid] = ipath

        frame_ids = sorted(unique_frames.keys())
        print(f"  Windows: {len(video_windows)}, unique frames: {len(frame_ids)}")

        # Load annotations and build bbox index
        ann_data = load_video_annotations(video_id)
        bbox_index = build_bbox_index(ann_data)
        print(f"  Bbox index: {len(bbox_index)} frames with own-hand annotations")

        # Process each frame
        frames_result = {}
        t_start = time.time()
        no_bbox_count = 0
        missing_side_count = 0

        for i, frame_id in enumerate(frame_ids):
            image_path = PROJECT_ROOT / unique_frames[frame_id]
            if not image_path.exists():
                print(f"  [{i+1}/{len(frame_ids)}] WARNING: missing {image_path}")
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  [{i+1}/{len(frame_ids)}] WARNING: could not read {image_path}")
                continue

            bboxes = bbox_index.get(frame_id, [])
            if not bboxes:
                no_bbox_count += 1
                continue

            t0 = time.time()
            pose = extract_frame_pose(pipe, image, bboxes)
            elapsed = time.time() - t0

            if pose:
                frames_result[frame_id] = pose
                sides = list(pose.keys())
                if len(sides) < 2:
                    missing_side_count += 1

            if (i + 1) % 50 == 0 or i == 0:
                elapsed_total = time.time() - t_start
                fps = (i + 1) / elapsed_total
                eta = (len(frame_ids) - i - 1) / fps if fps > 0 else 0
                print(f"  [{i+1}/{len(frame_ids)}] {frame_id}: {sides} ({elapsed:.2f}s) "
                      f"[{fps:.1f} frames/s, ETA {eta:.0f}s]")

            # Spot-check visualization
            if args.visualize and pose and (i % 50 == 0):
                vis_image = image.copy()
                for side, hand_data in pose.items():
                    vis_image = draw_keypoints(
                        vis_image, hand_data["keypoints_2d"], side == "right"
                    )
                vis_path = vis_dir / f"{frame_id}_keypoints.jpg"
                cv2.imwrite(str(vis_path), vis_image)

        elapsed_total = time.time() - t_start

        # Save results for this video
        output = {
            "metadata": {
                "video_id": video_id,
                "total_frames": len(frames_result),
                "mode": "crop_regress",
                "elapsed_seconds": round(elapsed_total, 1),
            },
            "frames": frames_result,
        }

        output_path = output_dir / f"video_{video_id}_poses.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        # Summary
        bimanual = sum(1 for f in frames_result.values() if "left" in f and "right" in f)
        print(f"\n  Video {video_id} complete in {elapsed_total:.1f}s")
        print(f"  Frames processed: {len(frames_result)}/{len(frame_ids)}")
        print(f"  Bimanual: {bimanual}/{len(frames_result)}")
        print(f"  No bbox: {no_bbox_count}, missing side: {missing_side_count}")
        print(f"  Saved to: {output_path}")

    print(f"\n{'=' * 60}")
    print("Step 2b complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
