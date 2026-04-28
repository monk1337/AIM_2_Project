"""
Step 1: Filter EgoSurgery frames for VLA prototype pipeline.

Filters:
  A) Phase filter, keep only incision, closure, dissection
  B) Hand visibility, keep only frames with hand bbox annotations
  C) Sample ~20 frames evenly across phases, preferring bimanual frames

Usage:
  python scripts/step1_filter_frames.py
  python scripts/step1_filter_frames.py --video 01 --samples-per-phase 7
  python scripts/step1_filter_frames.py --all-videos --samples-per-phase 5
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "full_data"

TARGET_PHASES = {"incision", "closure", "dissection"}


def load_phase_labels(video_id: str) -> dict[str, str]:
    """Load phase CSV and return {frame_id: phase} for target phases."""
    phase_dir = DATA_ROOT / "annotations" / "phase"
    frame_phases = {}

    # A video can have multiple segment CSVs (e.g., 01_1.csv, 01_2.csv)
    for csv_file in sorted(phase_dir.glob(f"{video_id}_*.csv")):
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_id = row["Frame"]
                phase = row["Phase"].strip().lower()
                if phase in TARGET_PHASES:
                    frame_phases[frame_id] = phase

    return frame_phases


def load_hand_annotations(video_id: str) -> dict[str, dict]:
    """Load hand bbox annotations and return {frame_id: {has_left, has_right, num_hands}}."""
    bbox_file = DATA_ROOT / "annotations" / "bbox" / "by_video" / "hands" / video_id / "annotations.json"

    if not bbox_file.exists():
        print(f"  WARNING: No hand bbox file for video {video_id}")
        return {}

    with open(bbox_file) as f:
        data = json.load(f)

    # Build image_id -> file_name mapping
    id_to_filename = {}
    for img in data["images"]:
        id_to_filename[img["id"]] = img["file_name"].replace(".jpg", "")

    # Build frame_id -> hand info
    frame_hands = {}
    for ann in data["annotations"]:
        frame_id = id_to_filename.get(ann["image_id"])
        if frame_id is None:
            continue

        if frame_id not in frame_hands:
            frame_hands[frame_id] = {
                "has_own_left": False,
                "has_own_right": False,
                "num_own_hands": 0,
            }

        cat_id = ann["category_id"]
        if cat_id == 1:  # Own hands left
            frame_hands[frame_id]["has_own_left"] = True
        elif cat_id == 2:  # Own hands right
            frame_hands[frame_id]["has_own_right"] = True

    # Count own hands per frame
    for frame_id, info in frame_hands.items():
        info["num_own_hands"] = int(info["has_own_left"]) + int(info["has_own_right"])

    return frame_hands


def load_gaze_data(video_id: str) -> dict[str, dict]:
    """Load gaze CSV and return {frame_id: {gaze_x, gaze_y}}."""
    gaze_dir = DATA_ROOT / "gaze"
    gaze_data = {}

    for csv_file in sorted(gaze_dir.glob(f"{video_id}_*.csv")):
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_id = row["Frame"]
                try:
                    gaze_data[frame_id] = {
                        "gaze_x": float(row["norm_pos_x"]),
                        "gaze_y": float(row["norm_pos_y"]),
                    }
                except (ValueError, KeyError):
                    continue

    return gaze_data


def filter_and_sample(video_id: str, samples_per_phase: int, seed: int = 42) -> list[dict]:
    """Run the full filtering pipeline for one video."""
    print(f"\n{'='*60}")
    print(f"Processing video {video_id}")
    print(f"{'='*60}")

    # Filter A: Phase filter
    phase_labels = load_phase_labels(video_id)
    print(f"  Phase filter: {len(phase_labels)} frames in target phases")
    for phase in TARGET_PHASES:
        count = sum(1 for p in phase_labels.values() if p == phase)
        print(f"    {phase}: {count}")

    if not phase_labels:
        print("  SKIPPED: No frames in target phases")
        return []

    # Filter B: Hand visibility
    hand_annotations = load_hand_annotations(video_id)
    print(f"  Hand annotations available: {len(hand_annotations)} frames")

    # Intersect: frames that pass both filters
    valid_frames = {}
    for frame_id, phase in phase_labels.items():
        if frame_id in hand_annotations:
            valid_frames[frame_id] = {
                "frame_id": frame_id,
                "video_id": video_id,
                "phase": phase,
                "image_path": str(
                    Path("full_data") / "images" / video_id / f"{frame_id}.jpg"
                ),
                **hand_annotations[frame_id],
            }

    print(f"  After both filters: {len(valid_frames)} frames")

    if not valid_frames:
        print("  SKIPPED: No frames pass both filters")
        return []

    # Load gaze data (optional enrichment)
    gaze_data = load_gaze_data(video_id)

    # Add gaze info to valid frames
    for frame_id, info in valid_frames.items():
        if frame_id in gaze_data:
            info["gaze_x"] = gaze_data[frame_id]["gaze_x"]
            info["gaze_y"] = gaze_data[frame_id]["gaze_y"]
        else:
            info["gaze_x"] = None
            info["gaze_y"] = None

    # Sample: prefer bimanual frames, sample evenly across phases
    random.seed(seed)
    selected = []

    for phase in sorted(TARGET_PHASES):
        phase_frames = [f for f in valid_frames.values() if f["phase"] == phase]
        if not phase_frames:
            print(f"  No frames for phase '{phase}' after filtering")
            continue

        # Sort: bimanual first (2 hands > 1 hand), then by frame order
        phase_frames.sort(key=lambda x: (-x["num_own_hands"], x["frame_id"]))

        # Take bimanual frames first, then fill with single-hand
        bimanual = [f for f in phase_frames if f["num_own_hands"] == 2]
        single = [f for f in phase_frames if f["num_own_hands"] == 1]

        n = min(samples_per_phase, len(phase_frames))

        if len(bimanual) >= n:
            # Enough bimanual, sample evenly spaced
            step = max(1, len(bimanual) // n)
            chosen = [bimanual[i * step] for i in range(n)]
        else:
            # Take all bimanual + fill from single
            chosen = bimanual.copy()
            remaining = n - len(chosen)
            if single and remaining > 0:
                step = max(1, len(single) // remaining)
                chosen += [single[i * step] for i in range(min(remaining, len(single)))]

        selected.extend(chosen)
        print(f"  Selected {len(chosen)} frames for '{phase}' ({sum(1 for c in chosen if c['num_own_hands']==2)} bimanual)")

    return selected


def main():
    parser = argparse.ArgumentParser(description="Step 1: Filter frames for VLA prototype")
    parser.add_argument("--video", default="01", help="Video ID to process (default: 01)")
    parser.add_argument("--all-videos", action="store_true", help="Process all videos")
    parser.add_argument("--samples-per-phase", type=int, default=7, help="Frames to sample per phase (default: 7)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    # Determine which videos to process
    if args.all_videos:
        video_ids = sorted([
            d.name for d in (DATA_ROOT / "images").iterdir() if d.is_dir()
        ])
    else:
        video_ids = [args.video]

    all_selected = []
    for vid in video_ids:
        selected = filter_and_sample(vid, args.samples_per_phase, args.seed)
        all_selected.extend(selected)

    # Summary
    print(f"\n{'='*60}")
    print(f"TOTAL SELECTED: {len(all_selected)} frames")
    for phase in sorted(TARGET_PHASES):
        count = sum(1 for s in all_selected if s["phase"] == phase)
        bimanual = sum(1 for s in all_selected if s["phase"] == phase and s["num_own_hands"] == 2)
        print(f"  {phase}: {count} ({bimanual} bimanual)")
    print(f"{'='*60}")

    # Save output
    output_dir = PROJECT_ROOT / "pipeline_output" / "step1_filtered"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output or str(output_dir / "selected_frames.json")
    with open(output_path, "w") as f:
        json.dump(all_selected, f, indent=2)
    print(f"\nSaved to: {output_path}")

    # Also save as CSV for easy viewing
    csv_path = output_path.replace(".json", ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_id", "video_id", "phase", "image_path",
            "has_own_left", "has_own_right", "num_own_hands",
            "gaze_x", "gaze_y",
        ])
        writer.writeheader()
        writer.writerows(all_selected)
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
