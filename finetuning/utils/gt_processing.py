"""
Shared GT computation utilities for POV-Surgery + WiLoR finetuning pipeline.
Extracted from AIM_2_Project/eval/run_eval_wilor.py.
"""

import pickle
import numpy as np
import torch
import cv2

# POV-Surgery camera intrinsics (fixed across all frames)
K = np.array([
    [1198.4395, 0.0, 960.0],
    [0.0, 1198.4395, 175.2],
    [0.0, 0.0, 1.0],
])

# OpenGL -> OpenCV coordinate flip
COORD_CHANGE = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])

# Joint reordering: MANO (16 base + 5 tips) -> OpenPose convention (21 joints)
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


def build_gt_mano(mano_model_dir, device="cpu"):
    """Build a MANO model for GT computation (flat_hand_mean=True, matching POV-Surgery annotations)."""
    import smplx
    from smplx.vertex_ids import vertex_ids as smplx_vertex_ids

    mano = smplx.create(
        str(mano_model_dir),
        model_type="mano",
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)

    tip_ids = [
        smplx_vertex_ids["mano"]["thumb"],
        smplx_vertex_ids["mano"]["index"],
        smplx_vertex_ids["mano"]["middle"],
        smplx_vertex_ids["mano"]["ring"],
        smplx_vertex_ids["mano"]["pinky"],
    ]

    return mano, tip_ids


def mano_forward(mano_model, tip_ids, global_orient, hand_pose, betas, device="cpu"):
    """Run MANO FK, return 21 joints (OpenPose order) and 778 vertices."""
    with torch.no_grad():
        out = mano_model(
            global_orient=torch.tensor(global_orient, dtype=torch.float32, device=device),
            hand_pose=torch.tensor(hand_pose, dtype=torch.float32, device=device),
            betas=torch.tensor(betas, dtype=torch.float32, device=device),
            transl=torch.zeros(1, 3, dtype=torch.float32, device=device),
        )
    joints_16 = out.joints[0].cpu().numpy()
    vertices = out.vertices[0].cpu().numpy()

    tips = vertices[tip_ids]
    joints_21 = np.concatenate([joints_16, tips])
    joints_21 = joints_21[MANO_TO_OPENPOSE]

    return joints_21, vertices


def transform_to_camera(joints, verts, anno):
    """Transform MANO joints/verts from local GRAB frame to OpenCV camera frame.

    Args:
        joints: (21, 3) joints in MANO local frame (OpenPose order)
        verts: (778, 3) vertices in MANO local frame
        anno: dict with keys cam_rot, cam_transl, grab2world_R, grab2world_T, transl

    Returns:
        joints_cv: (21, 3) in camera frame (OpenCV convention)
        verts_cv: (778, 3) in camera frame
    """
    cam_rot = anno["cam_rot"]
    cam_transl = anno["cam_transl"]
    g2w_R = anno["grab2world_R"]
    g2w_T = anno["grab2world_T"]
    transl = anno["transl"]

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = cam_transl
    camera_pose[:3, :3] = cam_rot
    cam_inv = np.linalg.inv(camera_pose)

    all_addition_g = g2w_R @ cam_inv[:3, :3].T
    all_addition_t = g2w_T @ cam_inv[:3, :3].T + cam_inv[:3, 3]
    all_addition_t_no_transl = transl @ all_addition_g + all_addition_t

    joints_cam = joints @ all_addition_g + all_addition_t_no_transl
    verts_cam = verts @ all_addition_g + all_addition_t_no_transl

    joints_cv = joints_cam @ COORD_CHANGE.T
    verts_cv = verts_cam @ COORD_CHANGE.T

    return joints_cv, verts_cv


def compute_global_orient_camera_frame(global_orient_aa, anno):
    """Transform global_orient from GRAB frame to camera frame.

    Args:
        global_orient_aa: (1, 3) or (3,) axis-angle in GRAB frame
        anno: dict with cam_rot, cam_transl, grab2world_R

    Returns:
        global_orient_cam: (3,) axis-angle in camera frame
    """
    go = np.array(global_orient_aa).flatten()
    R_go, _ = cv2.Rodrigues(go)  # (3, 3)

    cam_rot = anno["cam_rot"]
    cam_transl = anno["cam_transl"]
    g2w_R = anno["grab2world_R"]

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = cam_transl
    camera_pose[:3, :3] = cam_rot
    cam_inv = np.linalg.inv(camera_pose)

    # Full rotation chain: MANO local -> GRAB -> World -> Camera -> OpenCV
    R_total = COORD_CHANGE @ cam_inv[:3, :3] @ g2w_R
    R_go_cam = R_total @ R_go

    global_orient_cam, _ = cv2.Rodrigues(R_go_cam)
    return global_orient_cam.flatten().astype(np.float32)


def project_to_2d(joints_cam, K_mat=None):
    """Project 3D camera-frame joints to 2D pixel coordinates.

    Args:
        joints_cam: (N, 3) joints in camera frame
        K_mat: (3, 3) intrinsic matrix, defaults to POV-Surgery K

    Returns:
        joints_2d: (N, 2) pixel coordinates
    """
    if K_mat is None:
        K_mat = K
    proj = (K_mat @ joints_cam.T).T
    return proj[:, :2] / proj[:, 2:3]


def load_gt_annotation(pkl_path):
    """Load a single POV-Surgery annotation pickle."""
    with open(pkl_path, "rb") as f:
        anno = pickle.load(f)

    mano_params = anno["mano"]
    return {
        "global_orient": np.array(mano_params["global_orient"], dtype=np.float32),
        "hand_pose": np.array(mano_params["hand_pose"], dtype=np.float32),
        "betas": np.array(mano_params["betas"], dtype=np.float32),
        "transl": np.array(mano_params["transl"], dtype=np.float32),
        "cam_rot": np.array(anno.get("cam_rot", np.eye(3)), dtype=np.float64),
        "cam_transl": np.array(anno.get("cam_transl", np.zeros(3)), dtype=np.float64),
        "grab2world_R": np.array(anno.get("grab2world_R", np.eye(3)), dtype=np.float64),
        "grab2world_T": np.array(anno.get("grab2world_T", np.zeros((1, 3))), dtype=np.float64),
    }


def derive_bbox_from_2d_joints(joints_2d, img_shape=(1080, 1920), pad_factor=1.5):
    """Derive bounding box from 2D joint locations with padding.

    Args:
        joints_2d: (21, 2) pixel coordinates
        img_shape: (H, W)
        pad_factor: expansion factor around the joint bounding box

    Returns:
        bbox: (4,) array [x1, y1, x2, y2] or None if invalid
    """
    x_min, y_min = joints_2d.min(axis=0)
    x_max, y_max = joints_2d.max(axis=0)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    s = max(x_max - x_min, y_max - y_min) * pad_factor

    x1 = max(0, cx - s / 2)
    y1 = max(0, cy - s / 2)
    x2 = min(img_shape[1], cx + s / 2)
    y2 = min(img_shape[0], cy + s / 2)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ── Evaluation metrics ─────────────────────────────────────────────────

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
    """Mean Per-Joint Position Error (mm)."""
    return np.sqrt(((pred - gt) ** 2).sum(axis=-1)).mean() * 1000


def compute_pa_mpjpe(pred, gt):
    """Procrustes-Aligned MPJPE (mm)."""
    pred_aligned = procrustes_align(pred, gt)
    return np.sqrt(((pred_aligned - gt) ** 2).sum(axis=-1)).mean() * 1000


def compute_per_finger_mpjpe(pred, gt):
    """Per-finger MPJPE breakdown (mm), expects OpenPose-order joints."""
    pred_r = pred - pred[0:1]
    gt_r = gt - gt[0:1]
    fingers = {
        "thumb": slice(1, 5),
        "index": slice(5, 9),
        "middle": slice(9, 13),
        "ring": slice(13, 17),
        "pinky": slice(17, 21),
    }
    result = {}
    for name, sl in fingers.items():
        err = np.sqrt(((pred_r[sl] - gt_r[sl]) ** 2).sum(axis=-1)).mean() * 1000
        result[name] = err
    return result
