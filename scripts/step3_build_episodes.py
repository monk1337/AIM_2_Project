"""
Step 3: Build VITRA-compatible .npy episodes from step2b poses + language annotations.

Combines:
  - step1b windows (frame sequences per phase)
  - step2b poses (MANO params per frame)
  - Template-based language annotations
into VITRA episode .npy files matching EpisodicDatasetCore's expected format.

Usage:
  .venv/bin/python scripts/step3_build_episodes.py --video 01
  .venv/bin/python scripts/step3_build_episodes.py --all-videos
  .venv/bin/python scripts/step3_build_episodes.py --video 01 --verify
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Language templates (placeholder, upgrade to VLM later) ---
LANGUAGE_TEMPLATES = {
    "incision": {
        "left": "Stabilize tissue with forceps during incision.",
        "right": "Cut along the incision line with the scalpel.",
    },
    "closure": {
        "left": "Grip the tissue edge with forceps for suturing.",
        "right": "Drive the needle through tissue to close the wound.",
    },
    "dissection": {
        "left": "Retract tissue to expose the surgical plane.",
        "right": "Separate tissue layers along the dissection plane.",
    },
}

# --- Tobii Pro Glasses 2 estimated intrinsics ---
# ~82° horizontal FOV, 1920x1080 resolution
# fx = fy = W / (2 * tan(hfov/2)) ≈ 1920 / (2 * tan(41°)) ≈ 1066
TOBII_INTRINSICS = np.array([
    [1066.0, 0.0, 960.0],
    [0.0, 1066.0, 540.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)


def axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle vector(s) to rotation matrix/matrices.

    Args:
        aa: (3,) single vector or (N, 3) batch → (3, 3) or (N, 3, 3)
    """
    if aa.ndim == 1:
        return R.from_rotvec(aa).as_matrix()
    return R.from_rotvec(aa).as_matrix()


def build_episode(window: dict, poses: dict) -> dict | None:
    """Build a single VITRA episode dict from a window + per-frame poses.

    Returns None if too many frames are missing pose data.
    """
    frame_ids = window["frame_ids"]
    T = len(frame_ids)
    phase = window["phase"]

    # Check pose coverage
    missing = [fid for fid in frame_ids if fid not in poses]
    if len(missing) > T * 0.2:  # >20% missing → skip
        print(f"  WARNING: {window['window_id']} — {len(missing)}/{T} frames missing poses, skipping")
        return None

    # --- Collect per-frame data ---
    left_global_orient = np.zeros((T, 3, 3), dtype=np.float64)
    left_hand_pose = np.zeros((T, 15, 3, 3), dtype=np.float64)
    left_transl = np.zeros((T, 3), dtype=np.float64)
    left_joints = np.zeros((T, 21, 3), dtype=np.float64)
    left_betas_all = []
    left_kept = np.zeros(T, dtype=np.int64)

    right_global_orient = np.zeros((T, 3, 3), dtype=np.float64)
    right_hand_pose = np.zeros((T, 15, 3, 3), dtype=np.float64)
    right_transl = np.zeros((T, 3), dtype=np.float64)
    right_joints = np.zeros((T, 21, 3), dtype=np.float64)
    right_betas_all = []
    right_kept = np.zeros(T, dtype=np.int64)

    for t, fid in enumerate(frame_ids):
        frame_pose = poses.get(fid)
        if frame_pose is None:
            continue

        for side, arrays in [
            ("left", (left_global_orient, left_hand_pose, left_transl,
                      left_joints, left_betas_all, left_kept)),
            ("right", (right_global_orient, right_hand_pose, right_transl,
                       right_joints, right_betas_all, right_kept)),
        ]:
            go, hp, tr, jt, betas_list, kept = arrays
            hand = frame_pose.get(side)
            if hand is None:
                continue

            # global_orient: (3,) axis-angle → (3, 3) rotation matrix
            go[t] = axis_angle_to_rotmat(np.array(hand["global_orient"]))

            # hand_pose: (45,) axis-angle → (15, 3) → (15, 3, 3) rotation matrices
            hp_aa = np.array(hand["hand_pose"]).reshape(15, 3)
            hp[t] = R.from_rotvec(hp_aa).as_matrix()  # (15, 3, 3)

            # translation
            tr[t] = np.array(hand["cam_t_full"])

            # 3D keypoints
            jt[t] = np.array(hand["keypoints_3d"])

            # betas
            betas_list.append(np.array(hand["betas"]))

            # mark valid
            kept[t] = 1

    # Average betas across frames (should be near-constant for same surgeon)
    left_beta = np.mean(left_betas_all, axis=0) if left_betas_all else np.zeros(10, dtype=np.float64)
    right_beta = np.mean(right_betas_all, axis=0) if right_betas_all else np.zeros(10, dtype=np.float64)

    # --- Compute wrist (palm centroid: mean of joints 0,2,5,9,13,17) ---
    palm_indices = [0, 2, 5, 9, 13, 17]
    left_wrist = np.mean(left_joints[:, palm_indices, :], axis=1, keepdims=True).astype(np.float32)  # (T,1,3)
    right_wrist = np.mean(right_joints[:, palm_indices, :], axis=1, keepdims=True).astype(np.float32)

    # --- Camera-space = world-space (identity extrinsics) ---
    # So camspace arrays are identical to worldspace arrays
    left_joints_cam = left_joints.astype(np.float32)
    right_joints_cam = right_joints.astype(np.float32)

    # --- Language text ---
    templates = LANGUAGE_TEMPLATES.get(phase, LANGUAGE_TEMPLATES["dissection"])
    text = {
        "right": [(templates["right"], (0, T))],
        "left": [(templates["left"], (0, T))],
    }

    # --- Build episode dict ---
    episode = {
        # Video metadata
        "video_clip_id_segment": np.zeros(T, dtype=np.int64),
        "video_name": f"EgoSurgery_{window['video_id']}",
        "video_decode_frame": np.arange(T, dtype=np.int64),
        "extrinsics": np.tile(np.eye(4, dtype=np.float64), (T, 1, 1)),  # (T, 4, 4) identity
        "intrinsics": TOBII_INTRINSICS.copy(),  # (3, 3)

        # Annotation
        "anno_type": "right",
        "text": text,

        # Trajectory statistics (placeholder — computed from actual data)
        "avg_speed": np.float64(0.0),
        "total_rotvec_degree": np.float64(0.0),
        "total_transl_dist": np.float64(0.0),

        # Left hand
        "left": {
            "beta": left_beta,  # (10,)
            "global_orient_worldspace": left_global_orient,  # (T, 3, 3)
            "global_orient_camspace": left_global_orient.copy(),  # same (identity extrinsics)
            "hand_pose": left_hand_pose,  # (T, 15, 3, 3)
            "transl_worldspace": left_transl,  # (T, 3)
            "transl_camspace": left_transl.copy(),  # same
            "kept_frames": left_kept,  # (T,)
            "joints_worldspace": left_joints,  # (T, 21, 3)
            "joints_camspace": left_joints_cam,  # (T, 21, 3) float32
            "wrist": left_wrist,  # (T, 1, 3) float32
            "max_translation_movement": None,
            "max_wrist_rotation_movement": None,
            "max_finger_joint_angle_movement": None,
        },

        # Right hand
        "right": {
            "beta": right_beta,
            "global_orient_worldspace": right_global_orient,
            "global_orient_camspace": right_global_orient.copy(),
            "hand_pose": right_hand_pose,
            "transl_worldspace": right_transl,
            "transl_camspace": right_transl.copy(),
            "kept_frames": right_kept,
            "joints_worldspace": right_joints,
            "joints_camspace": right_joints_cam,
            "wrist": right_wrist,
            "max_translation_movement": None,
            "max_wrist_rotation_movement": None,
            "max_finger_joint_angle_movement": None,
        },
    }

    # --- Compute trajectory stats ---
    if np.any(right_kept):
        valid_mask = right_kept.astype(bool)
        valid_transl = right_transl[valid_mask]
        if len(valid_transl) > 1:
            diffs = np.diff(valid_transl, axis=0)
            speeds = np.linalg.norm(diffs, axis=1)
            episode["avg_speed"] = np.float64(np.mean(speeds))
            episode["total_transl_dist"] = np.float64(np.sum(speeds))

        valid_orient = right_global_orient[valid_mask]
        if len(valid_orient) > 1:
            total_deg = 0.0
            for i in range(len(valid_orient) - 1):
                R_rel = valid_orient[i + 1] @ valid_orient[i].T
                rotvec = R.from_matrix(R_rel).as_rotvec()
                total_deg += np.degrees(np.linalg.norm(rotvec))
            episode["total_rotvec_degree"] = np.float64(total_deg)

    return episode


def verify_episode(episode: dict, window_id: str) -> list[str]:
    """Verify episode has correct shapes and valid rotation matrices.

    Returns list of error messages (empty = all good).
    """
    errors = []
    T = len(episode["video_decode_frame"])

    # Required top-level keys
    required_keys = [
        "video_name", "video_decode_frame", "extrinsics", "intrinsics",
        "anno_type", "text", "left", "right",
    ]
    for key in required_keys:
        if key not in episode:
            errors.append(f"Missing key: {key}")

    # Shape checks
    shape_checks = [
        ("extrinsics", (T, 4, 4)),
        ("intrinsics", (3, 3)),
    ]
    for key, expected_shape in shape_checks:
        if key in episode:
            actual = episode[key].shape
            if actual != expected_shape:
                errors.append(f"{key}: expected {expected_shape}, got {actual}")

    # Per-hand shape checks
    for side in ["left", "right"]:
        if side not in episode:
            continue
        hand = episode[side]
        hand_checks = [
            ("beta", (10,)),
            ("global_orient_worldspace", (T, 3, 3)),
            ("hand_pose", (T, 15, 3, 3)),
            ("transl_worldspace", (T, 3)),
            ("joints_worldspace", (T, 21, 3)),
            ("kept_frames", (T,)),
        ]
        for key, expected_shape in hand_checks:
            if key not in hand:
                errors.append(f"{side}.{key}: missing")
            elif hasattr(hand[key], "shape") and hand[key].shape != expected_shape:
                errors.append(f"{side}.{key}: expected {expected_shape}, got {hand[key].shape}")

    # Rotation matrix validity (sample a few frames)
    for side in ["left", "right"]:
        if side not in episode:
            continue
        hand = episode[side]
        kept = hand["kept_frames"].astype(bool)
        valid_indices = np.where(kept)[0]
        if len(valid_indices) == 0:
            continue

        # Check a sample of valid frames
        check_indices = valid_indices[::max(1, len(valid_indices) // 5)]
        for idx in check_indices:
            # Global orient
            R_mat = hand["global_orient_worldspace"][idx]
            RtR = R_mat @ R_mat.T
            if not np.allclose(RtR, np.eye(3), atol=1e-4):
                errors.append(f"{side}.global_orient[{idx}]: not orthogonal (max err {np.max(np.abs(RtR - np.eye(3))): .6f})")
                break
            det = np.linalg.det(R_mat)
            if not np.isclose(det, 1.0, atol=1e-4):
                errors.append(f"{side}.global_orient[{idx}]: det={det:.6f} (expected 1.0)")
                break

            # Hand pose joints
            for j in range(15):
                R_j = hand["hand_pose"][idx, j]
                RtR_j = R_j @ R_j.T
                if not np.allclose(RtR_j, np.eye(3), atol=1e-4):
                    errors.append(f"{side}.hand_pose[{idx},{j}]: not orthogonal")
                    break
            else:
                continue
            break

    # Text format check
    text = episode.get("text", {})
    for side in ["left", "right"]:
        if side not in text:
            errors.append(f"text['{side}']: missing")
        elif not isinstance(text[side], list) or len(text[side]) == 0:
            errors.append(f"text['{side}']: expected non-empty list of (text, (start, end))")

    # Kept frames coverage
    left_valid = np.sum(episode["left"]["kept_frames"]) if "left" in episode else 0
    right_valid = np.sum(episode["right"]["kept_frames"]) if "right" in episode else 0
    if left_valid == 0:
        errors.append("left: no valid frames (all kept_frames = 0)")
    if right_valid == 0:
        errors.append("right: no valid frames (all kept_frames = 0)")

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Build VITRA .npy episodes from step2b poses")
    parser.add_argument("--video", default="01", help="Video ID (default: 01)")
    parser.add_argument("--all-videos", action="store_true", help="Process all videos")
    parser.add_argument("--windows-json",
                        default=str(PROJECT_ROOT / "pipeline_output" / "step1b_windows" / "windows.json"))
    parser.add_argument("--poses-dir",
                        default=str(PROJECT_ROOT / "pipeline_output" / "step2b_window_poses"))
    parser.add_argument("--verify", action="store_true", help="Run verification checks on output")
    args = parser.parse_args()

    # Load windows
    with open(args.windows_json) as f:
        windows_data = json.load(f)
    all_windows = windows_data["windows"]

    # Determine videos
    if args.all_videos:
        video_ids = sorted(set(w["video_id"] for w in all_windows))
    else:
        video_ids = [args.video]

    output_base = PROJECT_ROOT / "pipeline_output" / "step3_episodes"

    total_episodes = 0
    total_errors = 0

    for video_id in video_ids:
        print(f"\n{'=' * 60}")
        print(f"Building episodes for video {video_id}")
        print(f"{'=' * 60}")

        # Load poses
        poses_path = Path(args.poses_dir) / f"video_{video_id}_poses.json"
        if not poses_path.exists():
            print(f"  ERROR: Poses file not found: {poses_path}")
            print(f"  Run step2b first: .venv/bin/python scripts/step2b_extract_window_poses.py --video {video_id}")
            continue

        with open(poses_path) as f:
            poses_data = json.load(f)
        poses = poses_data["frames"]
        print(f"  Loaded {len(poses)} frame poses")

        # Filter windows for this video
        video_windows = [w for w in all_windows if w["video_id"] == video_id]
        print(f"  Windows: {len(video_windows)}")

        # Output directory
        output_dir = output_base / f"video_{video_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        episodes_built = 0
        for w in video_windows:
            episode = build_episode(w, poses)
            if episode is None:
                continue

            # Save
            # Use window_id as filename (sanitize for filesystem)
            filename = w["window_id"].replace("/", "_") + ".npy"
            out_path = output_dir / filename
            np.save(str(out_path), episode)
            episodes_built += 1

            # Verify if requested
            if args.verify:
                errs = verify_episode(episode, w["window_id"])
                if errs:
                    total_errors += len(errs)
                    print(f"  VERIFY FAIL: {w['window_id']}")
                    for e in errs:
                        print(f"    - {e}")
                else:
                    T = len(w["frame_ids"])
                    left_valid = int(np.sum(episode["left"]["kept_frames"]))
                    right_valid = int(np.sum(episode["right"]["kept_frames"]))
                    print(f"  OK: {w['window_id']} (T={T}, L={left_valid}/{T}, R={right_valid}/{T})")

        total_episodes += episodes_built
        print(f"\n  Video {video_id}: {episodes_built}/{len(video_windows)} episodes saved to {output_dir}")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"STEP 3 SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Output: {output_base}")
    if args.verify:
        if total_errors == 0:
            print(f"  Verification: ALL PASSED")
        else:
            print(f"  Verification: {total_errors} ERRORS found")
    else:
        print(f"  Run with --verify to check episode validity")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
