"""
Step 1b: Sample consecutive frame windows for VITRA fine-tuning.

VITRA requires temporal episode sequences (>=16 consecutive frames), not isolated
frames. This script finds all maximal consecutive runs of frames where:
  - The frame is in a target phase (incision, closure, dissection)
  - Both surgeon hands (own left + own right) have bbox annotations

Windows are output for downstream WiLoR processing and VITRA episode creation.

Usage:
  python scripts/step1b_sample_windows.py                          # video 01, min 16
  python scripts/step1b_sample_windows.py --all-videos             # all 15 videos
  python scripts/step1b_sample_windows.py --min-length 20          # longer windows only
  python scripts/step1b_sample_windows.py --sliding-window 16 --stride 8  # fixed-size mode
"""

import argparse
import json
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "full_data"

# Import step1 functions
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from step1_filter_frames import load_hand_annotations, load_phase_labels

TARGET_PHASES = {"incision", "closure", "dissection"}


def parse_frame_id(frame_id: str) -> tuple[str, str, int]:
    """Parse frame ID into (video_id, stream_id, frame_number)."""
    parts = frame_id.split("_")
    video_id = parts[0]
    stream_id = parts[1]
    frame_number = int(parts[2])
    return video_id, stream_id, frame_number


def find_consecutive_runs(frame_numbers: list[int]) -> list[list[int]]:
    """Find maximal consecutive runs in a sorted list of frame numbers."""
    if not frame_numbers:
        return []

    runs = []
    current_run = [frame_numbers[0]]

    for i in range(1, len(frame_numbers)):
        if frame_numbers[i] == frame_numbers[i - 1] + 1:
            current_run.append(frame_numbers[i])
        else:
            runs.append(current_run)
            current_run = [frame_numbers[i]]

    runs.append(current_run)
    return runs


def find_windows(video_id: str, min_length: int, sliding_window: int | None = None,
                 stride: int | None = None) -> list[dict]:
    """Find all consecutive frame windows for a video."""
    print(f"\n{'=' * 60}")
    print(f"Processing video {video_id}")
    print(f"{'=' * 60}")

    # Load annotations (reuse step1 functions)
    phase_labels = load_phase_labels(video_id)
    hand_annotations = load_hand_annotations(video_id)

    print(f"  Phase filter: {len(phase_labels)} frames in target phases")
    print(f"  Hand annotations: {len(hand_annotations)} frames")

    # Intersect: frames in target phases with BOTH own hands annotated
    bimanual_frames = {}
    for frame_id, phase in phase_labels.items():
        hand_info = hand_annotations.get(frame_id)
        if hand_info and hand_info["has_own_left"] and hand_info["has_own_right"]:
            bimanual_frames[frame_id] = phase

    print(f"  Bimanual in target phases: {len(bimanual_frames)} frames")

    if not bimanual_frames:
        print("  SKIPPED: No bimanual frames in target phases")
        return []

    # Group by (stream_id, phase)
    groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for frame_id, phase in bimanual_frames.items():
        _, stream_id, frame_number = parse_frame_id(frame_id)
        key = (stream_id, phase)
        if key not in groups:
            groups[key] = []
        groups[key].append((frame_number, frame_id))

    # Find consecutive runs within each group
    windows = []
    for (stream_id, phase), frames in sorted(groups.items()):
        frames.sort(key=lambda x: x[0])
        frame_numbers = [f[0] for f in frames]
        frame_id_map = {f[0]: f[1] for f in frames}

        runs = find_consecutive_runs(frame_numbers)

        for run in runs:
            if len(run) < min_length:
                continue

            if sliding_window is not None:
                # Fixed-size overlapping windows
                effective_stride = stride or sliding_window
                for start_idx in range(0, len(run) - sliding_window + 1, effective_stride):
                    sub_run = run[start_idx:start_idx + sliding_window]
                    first, last = sub_run[0], sub_run[-1]
                    window_id = f"{video_id}_{stream_id}_{phase}_{first:04d}-{last:04d}"
                    frame_ids = [frame_id_map[n] for n in sub_run]
                    windows.append({
                        "window_id": window_id,
                        "video_id": video_id,
                        "stream_id": stream_id,
                        "phase": phase,
                        "num_frames": len(sub_run),
                        "frame_ids": frame_ids,
                        "image_paths": [
                            str(Path("full_data") / "images" / video_id / f"{fid}.jpg")
                            for fid in frame_ids
                        ],
                    })
            else:
                # Whole run as one window
                first, last = run[0], run[-1]
                window_id = f"{video_id}_{stream_id}_{phase}_{first:04d}-{last:04d}"
                frame_ids = [frame_id_map[n] for n in run]
                windows.append({
                    "window_id": window_id,
                    "video_id": video_id,
                    "stream_id": stream_id,
                    "phase": phase,
                    "num_frames": len(run),
                    "frame_ids": frame_ids,
                    "image_paths": [
                        str(Path("full_data") / "images" / video_id / f"{fid}.jpg")
                        for fid in frame_ids
                    ],
                })

    # Print summary for this video
    total_frames = sum(w["num_frames"] for w in windows)
    print(f"  Windows found: {len(windows)} ({total_frames} total frames)")
    for phase in sorted(TARGET_PHASES):
        phase_windows = [w for w in windows if w["phase"] == phase]
        if phase_windows:
            phase_frames = sum(w["num_frames"] for w in phase_windows)
            lengths = [w["num_frames"] for w in phase_windows]
            print(f"    {phase}: {len(phase_windows)} windows, "
                  f"{phase_frames} frames (lengths: {min(lengths)}-{max(lengths)})")

    return windows


def main():
    parser = argparse.ArgumentParser(
        description="Step 1b: Sample consecutive frame windows for VITRA")
    parser.add_argument("--video", default="01", help="Video ID to process (default: 01)")
    parser.add_argument("--all-videos", action="store_true", help="Process all videos")
    parser.add_argument("--min-length", type=int, default=16,
                        help="Minimum window length in frames (default: 16)")
    parser.add_argument("--sliding-window", type=int, default=None,
                        help="Fixed window size (enables sliding window mode)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride for sliding window (default: same as window size)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.stride is not None and args.sliding_window is None:
        parser.error("--stride requires --sliding-window")

    # Determine which videos to process
    if args.all_videos:
        video_ids = sorted([
            d.name for d in (DATA_ROOT / "images").iterdir() if d.is_dir()
        ])
    else:
        video_ids = [args.video]

    all_windows = []
    for vid in video_ids:
        windows = find_windows(vid, args.min_length, args.sliding_window, args.stride)
        all_windows.extend(windows)

    # Build output
    total_frames = sum(w["num_frames"] for w in all_windows)
    output = {
        "metadata": {
            "min_window_length": args.min_length,
            "sliding_window": args.sliding_window,
            "stride": args.stride,
            "target_phases": sorted(TARGET_PHASES),
            "videos_processed": video_ids,
            "total_windows": len(all_windows),
            "total_frames": total_frames,
        },
        "windows": all_windows,
    }

    # Summary
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {len(all_windows)} windows, {total_frames} frames")
    for phase in sorted(TARGET_PHASES):
        phase_windows = [w for w in all_windows if w["phase"] == phase]
        if phase_windows:
            phase_frames = sum(w["num_frames"] for w in phase_windows)
            print(f"  {phase}: {len(phase_windows)} windows, {phase_frames} frames")
    print(f"{'=' * 60}")

    # Save output
    output_dir = PROJECT_ROOT / "pipeline_output" / "step1b_windows"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output or str(output_dir / "windows.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
