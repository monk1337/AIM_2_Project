"""Unified dataset loader for Aria val and POV-Surgery test (HF parquet schemas).

Both schemas have: image, sequence_name, frame_id, hand_side, image_wh, intrinsics,
                   global_orient, hand_pose, betas, transl, joints_3d, joints_2d, vertices.
Aria val also has: eval_joints_3d (Aria native MPS, in MPS-native order), has_eval_gt.

For Aria, the "primary" GT is eval_joints_3d (Aria native, OP order after remap, OP_VALID 20 joints).
For POV, the "primary" GT is joints_3d (synthetic native MANO; full 21 joints).
For both, joints_3d/joints_2d/vertices are MANO-order pseudo (Aria) or native (POV).
"""
import io
import json
import sys
import glob
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, "/workspace/code")
from eval_joint_orders import aria_mps_to_op  # noqa: E402


# Path defaults match the original training environment. Override at runtime
# with environment variables to point the loader at your own filesystem
# (see SETUP.md "Path overrides").
import os as _os
ARIA_VAL_DIR = _os.environ.get("AIM2_ARIA_VAL_DIR",  "/workspace/datasets/aria_val/data")
POV_DIR      = _os.environ.get("AIM2_POV_DIR",       "/workspace/datasets/pov_surgery/data")
SIDECAR_DIR  = _os.environ.get("AIM2_SIDECAR_DIR",   "/workspace/datasets/phase0_sidecars")


def _load_reject_keys() -> set:
    """Drop keys per the lab's canonical clean filter (reject + skip)."""
    R = json.load(open(f"{SIDECAR_DIR}/reject_keys_all_20260419.json"))
    return set(R.get("val_reject_keys", [])) | set(R.get("val_skip_keys", []))


def _load_mps_sidecar() -> dict:
    """Per-(seq,frame,hand) MPS GT, j2d_v2_disp (21x2 display frame, MANO order),
    j3d_v2_cam (21x3 camera frame), confidence, etc."""
    return json.load(open(f"{SIDECAR_DIR}/mps_v2_val_20260419.json"))


def ccw90(j2d: np.ndarray, W: int) -> np.ndarray:
    """Transform 2D points from raw sensor frame → display frame: (u,v) → (v, W-1-u)."""
    return np.stack([j2d[:, 1], (W - 1) - j2d[:, 0]], axis=1).astype(np.float32)


def load_aria_val(only_eval_gt: bool = True, apply_reject: bool = True, with_sidecar: bool = True):
    """Lab-canonical Aria val loader.

    apply_reject=True → drops 519 frames per reject_keys_all_20260419.json → ~2,333 instances (vs 3,190 raw).
    with_sidecar=True → attaches `aria_mps_2d_disp` from mps_v2_val_20260419.json (display-frame, MANO-ordered).
    """
    files = sorted(glob.glob(f"{ARIA_VAL_DIR}/validation-*.parquet"))
    drop = _load_reject_keys() if apply_reject else set()
    sidecar = _load_mps_sidecar() if with_sidecar else {}
    rows = []
    for f in files:
        df = pq.read_table(f).to_pandas()
        if only_eval_gt:
            df = df[df.has_eval_gt].reset_index(drop=True)
        if drop:
            df = df[~df.apply(lambda r: f"{r.sequence_name}/{r.frame_id}" in drop, axis=1)].reset_index(drop=True)
        for _, r in df.iterrows():
            ej3d_mps = np.asarray(r.eval_joints_3d, dtype=np.float32).reshape(21, 3)
            ej3d_op = aria_mps_to_op(ej3d_mps)
            K = np.asarray(r.intrinsics, dtype=np.float32).reshape(3, 3)
            W = int(r.image_width)
            joints_2d_raw = np.asarray(r.joints_2d, dtype=np.float32).reshape(21, 2)
            joints_2d_display = ccw90(joints_2d_raw, W)
            sk = f"{r.sequence_name}/{r.frame_id}/{r.hand_side}"
            mps_entry = sidecar.get(sk)
            mps_2d = (np.asarray(mps_entry["j2d_v2_disp"], dtype=np.float32) if mps_entry else None)
            mps_3d_cam = (np.asarray(mps_entry["j3d_v2_cam"], dtype=np.float32) if mps_entry else None)
            mps_conf = (float(mps_entry["confidence"]) if mps_entry else None)
            rows.append({
                "dataset": "aria_val",
                "sequence_name": r.sequence_name,
                "frame_id": int(r.frame_id),
                "hand_side": r.hand_side,
                "image": Image.open(io.BytesIO(r.image["bytes"])).convert("RGB"),
                "image_wh": (W, int(r.image_height)),
                "intrinsics": K,
                "is_fisheye": True,
                "native_joints_3d": np.asarray(r.joints_3d, dtype=np.float32).reshape(21, 3),
                "native_joints_2d": joints_2d_display,
                "native_vertices": np.asarray(r.vertices, dtype=np.float32).reshape(778, 3),
                "native_is_pseudo": True,
                "aria_eval_joints_3d_op": ej3d_op,
                # Sidecar (canonical, lab-published comparable)
                "aria_mps_2d_disp": mps_2d,        # (21,2) display frame, MANO-ordered
                "aria_mps_3d_cam": mps_3d_cam,     # (21,3) camera frame, MANO-ordered
                "aria_mps_conf": mps_conf,
            })
    return rows


def project_3d_to_2d_pinhole(joints_3d_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pinhole project (N,3) camera-frame joints through K → (N,2) image px."""
    proj = (K @ joints_3d_cam.T).T
    return (proj[:, :2] / np.maximum(proj[:, 2:3], 1e-6)).astype(np.float32)


def load_pov_test(stride: int = 1, lazy_image: bool = True):
    """Stride-sampled POV-Surgery test.

    lazy_image=True: store image_bytes only; decode PIL on demand. Saves ~28GB RAM on full set.
    """
    files = sorted(glob.glob(f"{POV_DIR}/test-*.parquet"))
    rows = []
    if stride > 1:
        import pandas as pd
        meta_dfs = [pq.read_table(f, columns=["sequence_name", "frame_id"]).to_pandas() for f in files]
        meta = pd.concat(meta_dfs, ignore_index=True)
        keep_set = set()
        for seq, sub in meta.groupby("sequence_name"):
            keep_set.update((seq, int(fid)) for fid in sorted(sub.frame_id.unique())[::stride])
    for f in files:
        df = pq.read_table(f).to_pandas()
        if stride > 1:
            df = df[df.apply(lambda r: (r.sequence_name, int(r.frame_id)) in keep_set, axis=1)].reset_index(drop=True)
        for _, r in df.iterrows():
            K = np.asarray(r.intrinsics, dtype=np.float32).reshape(3, 3)
            row = {
                "dataset": "pov_test",
                "sequence_name": r.sequence_name,
                "frame_id": int(r.frame_id),
                "hand_side": r.hand_side,
                "image_wh": (int(r.image_width), int(r.image_height)),
                "intrinsics": K,
                "is_fisheye": False,
                "native_joints_3d": np.asarray(r.joints_3d, dtype=np.float32).reshape(21, 3),
                "native_joints_2d": np.asarray(r.joints_2d, dtype=np.float32).reshape(21, 2),
                "native_vertices": np.asarray(r.vertices, dtype=np.float32).reshape(778, 3),
                "native_is_pseudo": False,
                "aria_eval_joints_3d_op": None,
            }
            if lazy_image:
                row["_image_bytes"] = r.image["bytes"]
            else:
                row["image"] = Image.open(io.BytesIO(r.image["bytes"])).convert("RGB")
            rows.append(row)
    return rows


def get_image(row):
    """Decode PIL image lazily, supports both eager `image` and lazy `_image_bytes`."""
    if "image" in row and row["image"] is not None:
        return row["image"]
    return Image.open(io.BytesIO(row["_image_bytes"])).convert("RGB")


def derive_bbox_from_joints2d(joints_2d: np.ndarray, padding: float = 1.5,
                              img_wh: tuple = (1408, 1408),
                              size_min: float = 260.0, size_max: float = 1200.0) -> np.ndarray:
    """Canonical lab bbox: 1.5× max axis-range, square, clamped to [260, 1200] display-frame px."""
    x0, y0 = joints_2d.min(0)
    x1, y1 = joints_2d.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = padding * max(x1 - x0, y1 - y0)
    side = max(size_min, min(size_max, side))
    x = max(0, cx - side / 2)
    y = max(0, cy - side / 2)
    w = min(img_wh[0] - x, side)
    h = min(img_wh[1] - y, side)
    return np.array([x, y, w, h], dtype=np.float32)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["aria_val", "pov_test"], default="aria_val")
    p.add_argument("--limit", type=int, default=3)
    args = p.parse_args()
    rows = load_aria_val() if args.dataset == "aria_val" else load_pov_test()
    print(f"Loaded {len(rows)} rows from {args.dataset}")
    for r in rows[:args.limit]:
        print(f"  {r['sequence_name']}/{r['frame_id']} {r['hand_side']}  fisheye={r['is_fisheye']}  "
              f"pseudo_native={r['native_is_pseudo']}  K_diag={r['intrinsics'][0,0]:.0f}")
