"""
Visualize HaMeR predictions vs GT on sampled POV-Surgery test frames.

For each sampled frame, generates a 2x2 panel:
  [Top-left]     Original image with GT 2D joints
  [Top-right]    Original image with predicted 2D joints
  [Bottom-left]  GT mesh rendered and overlaid on image
  [Bottom-right] Predicted mesh rendered and overlaid on image

Usage:
  python visualize_hamer.py --data_dir /path/to/POV_Surgery_data --n_samples 10
  python visualize_hamer.py --data_dir /path/to/POV_Surgery_data --samples m_diskplacer_1/00063,m_diskplacer_1/00100
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # must be before pyrender import

import sys
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import pyrender
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HAMER_ROOT = str(Path(__file__).resolve().parent.parent / "hamer")
SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, HAMER_ROOT)

# MANO-order hand skeleton: (parent, child) pairs for drawing bones.
# MANO kinematic tree joint order (NOTE: pinky comes BEFORE ring!):
#   0: Wrist
#   1-3: Index (MCP, PIP, DIP)
#   4-6: Middle (MCP, PIP, DIP)
#   7-9: Pinky (MCP, PIP, DIP)    <-- Pinky before Ring in MANO!
#   10-12: Ring (MCP, PIP, DIP)
#   13-15: Thumb (CMC, MCP, IP)
#   16: Thumb tip, 17: Index tip, 18: Middle tip, 19: Ring tip, 20: Pinky tip
HAND_EDGES_MANO = [
    (0, 13), (13, 14), (14, 15), (15, 16),  # thumb
    (0, 1), (1, 2), (2, 3), (3, 17),        # index
    (0, 4), (4, 5), (5, 6), (6, 18),        # middle
    (0, 10), (10, 11), (11, 12), (12, 19),  # ring  (joints 10-12 -> ringtip 19)
    (0, 7), (7, 8), (8, 9), (9, 20),        # pinky (joints 7-9 -> pinkytip 20)
]

# Edge-to-finger mapping for coloring (indices into HAND_EDGES_MANO)
EDGE_FINGER = ['thumb'] * 4 + ['index'] * 4 + ['middle'] * 4 + ['ring'] * 4 + ['pinky'] * 4

JOINT_COLORS_MANO = [
    (255, 255, 255),  # 0  wrist - white
    *[(100, 255, 100)] * 3,   # 1-3   index - green
    *[(100, 100, 255)] * 3,   # 4-6   middle - blue
    *[(255, 100, 255)] * 3,   # 7-9   pinky - magenta (pinky before ring in MANO!)
    *[(255, 255, 100)] * 3,   # 10-12 ring - yellow
    *[(255, 100, 100)] * 3,   # 13-15 thumb - red
    (255, 100, 100),          # 16 thumb tip
    (100, 255, 100),          # 17 index tip
    (100, 100, 255),          # 18 middle tip
    (255, 255, 100),          # 19 ring tip
    (255, 100, 255),          # 20 pinky tip
]

FINGER_COLORS = {
    'thumb':  (255, 100, 100),
    'index':  (100, 255, 100),
    'middle': (100, 100, 255),
    'ring':   (255, 255, 100),
    'pinky':  (255, 100, 255),
}


def draw_joints_on_image(img, joints_2d, label, color_gt=True):
    """Draw 2D hand joints and skeleton on image. Joints assumed in MANO order."""
    vis = img.copy()
    h, w = vis.shape[:2]
    joints = joints_2d.astype(int)

    # Draw bones (MANO order)
    for i, (p, c) in enumerate(HAND_EDGES_MANO):
        color = FINGER_COLORS[EDGE_FINGER[i]]
        pt1 = tuple(joints[p].tolist())
        pt2 = tuple(joints[c].tolist())
        if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
            0 <= pt2[0] < w and 0 <= pt2[1] < h):
            cv2.line(vis, pt1, pt2, color, 2, cv2.LINE_AA)

    # Draw joints
    for i, (x, y) in enumerate(joints):
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(vis, (x, y), 4, JOINT_COLORS_MANO[i], -1, cv2.LINE_AA)
            cv2.circle(vis, (x, y), 4, (0, 0, 0), 1, cv2.LINE_AA)

    # Label
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def render_mesh_on_image(img, verts_3d, faces, camera_K, label, color=(0.5, 0.7, 1.0, 0.6)):
    """
    Render a 3D hand mesh overlaid on the image using pyrender.

    Args:
        verts_3d: (778, 3) vertices in camera frame (NOT root-centered)
        faces: (F, 3) mesh faces
        camera_K: (3, 3) camera intrinsic matrix
        color: RGBA color for the mesh
    """
    h, w = img.shape[:2]

    # Create mesh
    mesh = trimesh.Trimesh(vertices=verts_3d, faces=faces, process=False)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=color,
        metallicFactor=0.1,
        roughnessFactor=0.8,
        alphaMode='BLEND',
    )
    mesh_node = pyrender.Mesh.from_trimesh(mesh, material=material)

    # Create scene
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.5, 0.5, 0.5])
    scene.add(mesh_node)

    # Camera
    fx, fy = camera_K[0, 0], camera_K[1, 1]
    cx, cy = camera_K[0, 2], camera_K[1, 2]
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy,
                                        znear=0.01, zfar=10.0)
    scene.add(camera, pose=np.eye(4))

    # Light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=np.eye(4))

    # Render
    renderer = pyrender.OffscreenRenderer(w, h)
    rendered, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    # Composite onto image
    alpha = rendered[:, :, 3:4].astype(float) / 255.0
    vis = img.astype(float) * (1 - alpha) + rendered[:, :, :3].astype(float) * alpha
    vis = vis.astype(np.uint8)

    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def compute_gt_cam_verts(annotation, gt_mano_layer, device):
    """
    Compute GT vertices in camera frame (NOT root-centered) for rendering.
    Returns (778, 3) numpy array in meters.
    """
    from evaluate_hamer import axis_angle_to_rotmat, COORD_CHANGE_MAT

    mano_params = annotation["mano"]
    global_orient_aa = torch.tensor(mano_params["global_orient"], dtype=torch.float32, device=device)
    hand_pose_aa = torch.tensor(mano_params["hand_pose"], dtype=torch.float32, device=device)
    betas = torch.tensor(mano_params["betas"], dtype=torch.float32, device=device)
    transl = mano_params["transl"].astype(np.float32)

    cam_rot = annotation["cam_rot"].astype(np.float32)
    cam_transl = annotation["cam_transl"].astype(np.float32)
    grab2world_R = annotation["grab2world_R"].astype(np.float32)
    grab2world_T = annotation["grab2world_T"].astype(np.float32)

    global_orient = axis_angle_to_rotmat(global_orient_aa)
    hand_pose = axis_angle_to_rotmat(hand_pose_aa.reshape(-1, 3))
    hand_pose = hand_pose.squeeze(1).unsqueeze(0)

    with torch.no_grad():
        mano_out = gt_mano_layer(global_orient=global_orient, hand_pose=hand_pose, betas=betas)

    gt_verts = mano_out.vertices
    gt_joints = mano_out.joints

    cam_transl_flat = cam_transl.flatten()
    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[:3, :3] = cam_rot
    camera_pose[:3, 3] = cam_transl_flat
    inv_cam = np.linalg.inv(camera_pose)

    all_addition_g = grab2world_R @ inv_cam[:3, :3].T
    all_addition_t = grab2world_T @ inv_cam[:3, :3].T + inv_cam[:3, 3:].T
    all_addition_t_no_transl = transl @ all_addition_g + all_addition_t

    all_addition_g_t = torch.tensor(all_addition_g, dtype=torch.float32, device=device)
    all_addition_t_no_transl_t = torch.tensor(all_addition_t_no_transl, dtype=torch.float32, device=device)
    coord_change = torch.tensor(COORD_CHANGE_MAT, dtype=torch.float32, device=device)

    gt_verts = gt_verts @ all_addition_g_t + all_addition_t_no_transl_t
    gt_verts = gt_verts @ coord_change.T

    gt_joints = gt_joints @ all_addition_g_t + all_addition_t_no_transl_t
    gt_joints = gt_joints @ coord_change.T

    return gt_verts[0].cpu().numpy(), gt_joints[0].cpu().numpy()


def run_hamer_on_frame(model, model_cfg, img, bbox, device):
    """
    Run HaMeR and return raw (non-root-centered) outputs for visualization.
    Returns pred_verts_cam (778, 3), pred_joints_cam (21, 3), pred_2d (21, 2).
    """
    import io
    from contextlib import redirect_stdout
    from hamer.datasets.vitdet_dataset import ViTDetDataset

    with redirect_stdout(io.StringIO()):
        dataset = ViTDetDataset(
            model_cfg, img, boxes=bbox.reshape(1, 4),
            right=np.array([1.0]), rescale_factor=2.5,
        )
        item = dataset[0]

    batch = {}
    for k, v in item.items():
        if isinstance(v, np.ndarray):
            batch[k] = torch.tensor(v, device=device).unsqueeze(0).float()
        elif isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, (int, float)):
            batch[k] = torch.tensor([v], device=device).float()
        else:
            batch[k] = v

    with torch.no_grad():
        out = model(batch)

    pred_verts = out["pred_vertices"][0].cpu().numpy()    # (778, 3)
    pred_joints = out["pred_keypoints_3d"][0].cpu().numpy()  # (21, 3) OpenPose order
    pred_2d = out["pred_keypoints_2d"][0].cpu().numpy()   # (21, 2) normalized, OpenPose order
    pred_cam_t = out["pred_cam_t"][0].cpu().numpy()       # (3,) camera translation

    # Reorder from OpenPose to MANO order (to match GT and skeleton drawing)
    from evaluate_hamer import OPENPOSE_TO_MANO
    pred_joints = pred_joints[OPENPOSE_TO_MANO, :]
    pred_2d = pred_2d[OPENPOSE_TO_MANO, :]

    # Map 2D to pixel coords
    box_center = item["box_center"]
    box_size = item["box_size"]
    pred_2d_pixels = pred_2d * box_size + box_center  # (21, 2)

    # Shift verts/joints to camera frame using predicted translation
    pred_verts_cam = pred_verts + pred_cam_t[None, :]
    pred_joints_cam = pred_joints + pred_cam_t[None, :]

    return pred_verts_cam, pred_joints_cam, pred_2d_pixels


def create_visualization(img, gt_joints_2d, pred_joints_2d,
                         gt_verts_cam, pred_verts_cam, faces, camera_K,
                         sample_key, output_path):
    """Create 2x2 panel visualization and save."""
    # Top row: 2D joints
    panel_joints_gt = draw_joints_on_image(img, gt_joints_2d, "GT 2D Joints")
    panel_joints_pred = draw_joints_on_image(img, pred_joints_2d, "Pred 2D Joints")

    # Bottom row: mesh overlay
    if gt_verts_cam is not None:
        panel_mesh_gt = render_mesh_on_image(
            img, gt_verts_cam, faces, camera_K,
            "GT Mesh", color=(0.3, 0.8, 0.3, 0.6))  # green
    else:
        panel_mesh_gt = img.copy()
        cv2.putText(panel_mesh_gt, "GT Mesh (N/A)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    panel_mesh_pred = render_mesh_on_image(
        img, pred_verts_cam, faces, camera_K,
        "Pred Mesh", color=(0.5, 0.5, 1.0, 0.6))  # blue

    # Assemble 2x2 grid
    top = np.concatenate([panel_joints_gt, panel_joints_pred], axis=1)
    bottom = np.concatenate([panel_mesh_gt, panel_mesh_pred], axis=1)
    panel = np.concatenate([top, bottom], axis=0)

    # Add title
    title_bar = np.zeros((40, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, sample_key, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    panel = np.concatenate([title_bar, panel], axis=0)

    cv2.imwrite(output_path, panel)
    return panel


def main():
    parser = argparse.ArgumentParser(description="Visualize HaMeR vs GT on POV-Surgery")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of random samples to visualize")
    parser.add_argument("--samples", type=str, default=None,
                        help="Comma-separated sample keys (e.g. m_diskplacer_1/00063,m_diskplacer_1/00100)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: AIM_2_Project/visualizations)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.data_dir = os.path.abspath(args.data_dir)
    output_dir = args.output_dir or os.path.join(SCRIPT_DIR, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.chdir(HAMER_ROOT)

    # Camera intrinsics (POV-Surgery)
    camera_K = np.array([
        [1198.4395, 0., 960.0],
        [0., 1198.4395, 175.2],
        [0., 0., 1.],
    ], dtype=np.float32)

    # Load test split
    sys.path.insert(0, SCRIPT_DIR)
    from evaluate_hamer import load_test_split, load_frame_data, get_bbox_from_joints, create_gt_mano_layer

    samples = load_test_split(args.data_dir)
    sample_dict = {f"{s}/{f}": (s, f, j) for s, f, j in samples}

    # Select samples
    if args.samples:
        selected_keys = [k.strip() for k in args.samples.split(",")]
    else:
        np.random.seed(args.seed)
        keys = list(sample_dict.keys())
        selected_keys = list(np.random.choice(keys, size=min(args.n_samples, len(keys)), replace=False))

    print(f"Visualizing {len(selected_keys)} samples -> {output_dir}")

    # Load models
    from hamer.models import load_hamer, download_models, DEFAULT_CHECKPOINT
    from hamer.configs import CACHE_DIR_HAMER

    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()

    mano_dir = os.path.join(HAMER_ROOT, "_DATA", "data", "mano")
    gt_mano = create_gt_mano_layer(mano_dir, device)

    # Get MANO faces
    faces = gt_mano.faces.astype(np.int32)

    for i, key in enumerate(selected_keys):
        if key not in sample_dict:
            print(f"  [{i+1}] {key} - not found in test split, skipping")
            continue

        seq_name, frame_id, joints_uv = sample_dict[key]
        img, annotation = load_frame_data(args.data_dir, seq_name, frame_id)
        if img is None or annotation is None:
            print(f"  [{i+1}] {key} - missing data, skipping")
            continue

        try:
            # GT mesh in camera frame (not root-centered, for rendering)
            gt_verts_cam, gt_joints_cam = compute_gt_cam_verts(annotation, gt_mano, device)

            # HaMeR prediction
            bbox = get_bbox_from_joints(joints_uv, expansion_factor=1.5)
            pred_verts_cam, pred_joints_cam, pred_2d = run_hamer_on_frame(
                model, model_cfg, img, bbox, device)

            # Save visualization
            safe_key = key.replace("/", "_")
            out_path = os.path.join(output_dir, f"{safe_key}.jpg")
            create_visualization(
                img, joints_uv, pred_2d,
                gt_verts_cam, pred_verts_cam, faces, camera_K,
                key, out_path)

            print(f"  [{i+1}/{len(selected_keys)}] {key} -> {out_path}")

        except Exception as e:
            print(f"  [{i+1}/{len(selected_keys)}] {key} - FAILED: {e}")

    print(f"\nDone! Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
