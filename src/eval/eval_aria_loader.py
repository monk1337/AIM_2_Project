"""Aria val loader with proper joint-order handling.

For each row:
  - eval_joints_3d_op: Aria native MPS GT remapped to OpenPose order (21x3, OP[1] is dummy zero)
  - eval_joints_2d_op: projected Aria 3D → 2D via GT intrinsics (21x2, OP[1] dummy)
  - hamersam_joints_3d_mano: HaMeR pseudo-label 3D in MANO order (21x3)
  - hamersam_joints_2d: HaMeR pseudo-label 2D in MANO order (21x2)
  - hamersam_vertices: HaMeR pseudo-label MANO mesh (778x3)
  - intrinsics: 3x3 K
  - image, image_wh, hand_side, sequence_name, frame_id
"""
import io
import sys
import glob
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, "/workspace/code")
from eval_joint_orders import aria_mps_to_op  # noqa: E402


ARIA_VAL_DIR = "/workspace/datasets/aria_val/data"


def project_3d_to_2d(joints_3d_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project 3D points (N,3) in camera frame through intrinsics K (3,3) → (N,2) pixels."""
    pts = (K @ joints_3d_cam.T).T
    pts = pts[:, :2] / np.maximum(pts[:, 2:3], 1e-6)
    return pts


def load_aria_val(only_eval_gt: bool = True):
    files = sorted(glob.glob(f"{ARIA_VAL_DIR}/validation-*.parquet"))
    rows = []
    for f in files:
        df = pq.read_table(f).to_pandas()
        if only_eval_gt:
            df = df[df.has_eval_gt].reset_index(drop=True)
        for _, r in df.iterrows():
            ej3d_mps = np.asarray(r.eval_joints_3d, dtype=np.float32).reshape(21, 3)
            ej3d_op = aria_mps_to_op(ej3d_mps)
            K = np.asarray(r.intrinsics, dtype=np.float32).reshape(3, 3)
            ej2d_op = project_3d_to_2d(ej3d_op, K)
            rows.append({
                "sequence_name": r.sequence_name,
                "frame_id": int(r.frame_id),
                "hand_side": r.hand_side,
                "image": Image.open(io.BytesIO(r.image["bytes"])).convert("RGB"),
                "image_wh": (int(r.image_width), int(r.image_height)),
                "intrinsics": K,
                # HaMeRSAM pseudo (MANO order)
                "hamersam_joints_3d_mano": np.asarray(r.joints_3d, dtype=np.float32).reshape(21, 3),
                "hamersam_joints_2d_mano": np.asarray(r.joints_2d, dtype=np.float32).reshape(21, 2),
                "hamersam_vertices": np.asarray(r.vertices, dtype=np.float32).reshape(778, 3),
                # Aria native (OP order, OP[1] dummy)
                "eval_joints_3d_mps_native": ej3d_mps,
                "eval_joints_3d_op": ej3d_op,
                "eval_joints_2d_op": ej2d_op,
            })
    return rows


def derive_bbox_from_joints2d(joints_2d: np.ndarray, padding: float = 1.5,
                              img_wh: tuple = (1408, 1408)) -> np.ndarray:
    x0, y0 = joints_2d.min(0)
    x1, y1 = joints_2d.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * padding
    side = max(side, 64)
    x = max(0, cx - side / 2)
    y = max(0, cy - side / 2)
    w = min(img_wh[0] - x, side)
    h = min(img_wh[1] - y, side)
    return np.array([x, y, w, h], dtype=np.float32)


if __name__ == "__main__":
    rows = load_aria_val()
    print(f"Loaded {len(rows)} Aria val instances")
    r = rows[0]
    print("OP order eval joints (first 5):\n", r["eval_joints_3d_op"][:5])
    print("MANO order HaMeRSAM joints (first 5):\n", r["hamersam_joints_3d_mano"][:5])
    print("Aria 2D projected (first 5):\n", r["eval_joints_2d_op"][:5])
    print("HaMeRSAM 2D (first 5):\n", r["hamersam_joints_2d_mano"][:5])
