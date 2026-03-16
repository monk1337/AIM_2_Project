"""
Evaluate pretrained HaMeR (zero-shot) on POV-Surgery test set.

Computes the same 5 metrics as Table 2 in the MICCAI paper:
  P_2d     - 2D joint reprojection error (pixels)
  MPJPE    - Mean Per-Joint Position Error (mm)
  PVE      - Per-Vertex Error (mm)
  PA-MPJPE - Procrustes-Aligned MPJPE (mm)
  PA-PVE   - Procrustes-Aligned PVE (mm)

Usage:
  # Step 1: Precompute GT (one-time, reusable across models)
  python evaluate_hamer.py --data_dir /path/to/POV_Surgery_data --precompute_gt

  # Step 2: Evaluate HaMeR (loads cached GT)
  python evaluate_hamer.py --data_dir /path/to/POV_Surgery_data
"""

import os
import sys
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add HaMeR to path
HAMER_ROOT = str(Path(__file__).resolve().parent.parent / "hamer")
SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, HAMER_ROOT)

GT_CACHE_FILENAME = "gt_cache_povsurgery_test.pkl"


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_similarity_transform(S1, S2):
    """
    Procrustes alignment: find sR, t that aligns S1 to S2.
    Args:
        S1: (B, N, 3) predicted points
        S2: (B, N, 3) ground truth points
    Returns:
        S1_hat: (B, N, 3) aligned predicted points
    """
    from hamer.utils.pose_utils import compute_similarity_transform as _cst
    return _cst(S1, S2)


def compute_metrics(pred_joints, gt_joints, pred_verts, gt_verts):
    """
    Compute MPJPE, PVE, PA-MPJPE, PA-PVE for a single sample.
    All inputs are root-centered tensors of shape (1, N, 3) in meters.
    Returns dict with values in mm.
    """
    mpjpe = torch.sqrt(((pred_joints - gt_joints) ** 2).sum(-1)).mean(-1) * 1000
    pve = torch.sqrt(((pred_verts - gt_verts) ** 2).sum(-1)).mean(-1) * 1000

    pred_j_aligned = compute_similarity_transform(pred_joints, gt_joints)
    pa_mpjpe = torch.sqrt(((pred_j_aligned - gt_joints) ** 2).sum(-1)).mean(-1) * 1000

    pred_v_aligned = compute_similarity_transform(pred_verts, gt_verts)
    pa_pve = torch.sqrt(((pred_v_aligned - gt_verts) ** 2).sum(-1)).mean(-1) * 1000

    return {
        "mpjpe": mpjpe.item(),
        "pve": pve.item(),
        "pa_mpjpe": pa_mpjpe.item(),
        "pa_pve": pa_pve.item(),
    }


# ─── GT computation ──────────────────────────────────────────────────────────

def create_gt_mano_layer(mano_model_path, device):
    """
    Create a MANO layer for GT computation.
    Uses plain smplx.MANOLayer (NO OpenPose reordering) with flat_hand_mean=True
    to match POV-Surgery's GT annotation convention.

    smplx.MANOLayer outputs 16 base joints in MANO order. We add 5 fingertips
    from mesh vertices externally in compute_gt_3d().
    """
    import smplx
    gt_mano = smplx.MANOLayer(
        model_path=mano_model_path,
        is_rhand=True,
        flat_hand_mean=True,
        use_pca=False,
        num_hand_joints=15,
    ).to(device)
    gt_mano.eval()
    return gt_mano


# OpenPose-to-MANO joint order mapping.
# HaMeR outputs 21 joints in OpenPose order via mano_to_openpose reordering.
# GT joints are in MANO order: 16 base joints + 5 fingertips (thumb, index, middle, ring, pinky).
# To compare, we reorder HaMeR's OpenPose joints back to MANO order.
# mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
# The inverse mapping (openpose_to_mano):
OPENPOSE_TO_MANO = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]


COORD_CHANGE_MAT = np.array([[1., 0., 0.],
                               [0., -1., 0.],
                               [0., 0., -1.]], dtype=np.float32)


def axis_angle_to_rotmat(aa):
    """
    Convert axis-angle to rotation matrix using cv2.Rodrigues.
    Args:
        aa: (B, 3) tensor of axis-angle vectors
    Returns:
        rotmat: (B, 1, 3, 3) tensor of rotation matrices
    """
    batch = aa.shape[0]
    aa_np = aa.detach().cpu().numpy()
    rotmats = np.zeros((batch, 3, 3), dtype=np.float32)
    for i in range(batch):
        rotmats[i], _ = cv2.Rodrigues(aa_np[i])
    return torch.tensor(rotmats, dtype=aa.dtype, device=aa.device).unsqueeze(1)


def compute_gt_3d(annotation, gt_mano_layer, device):
    """
    Compute root-centered GT 3D joints and vertices in camera frame.

    Pipeline from POV_Surgery/HandOccNet_ft/common/nets/mano_head.py lines 220-253:
      1. MANO forward pass (flat_hand_mean=True)
      2. grab2world + camera transform
      3. OpenGL->OpenCV coordinate change
      4. Root-center (wrist)

    Returns:
        gt_joints: (21, 3) numpy array in meters, root-centered
        gt_verts:  (778, 3) numpy array in meters, root-centered
    """
    mano_params = annotation["mano"]
    # smplx 0.1.28 MANOLayer expects rotation matrices (pose2rot=False),
    # but POV-Surgery stores axis-angle. Convert here.
    global_orient_aa = torch.tensor(mano_params["global_orient"], dtype=torch.float32, device=device)  # (1, 3)
    hand_pose_aa = torch.tensor(mano_params["hand_pose"], dtype=torch.float32, device=device)  # (1, 45)
    betas = torch.tensor(mano_params["betas"], dtype=torch.float32, device=device)
    transl = mano_params["transl"].astype(np.float32)  # (1, 3)

    # Convert axis-angle to rotation matrices
    global_orient = axis_angle_to_rotmat(global_orient_aa)  # (1, 1, 3, 3)
    hand_pose = axis_angle_to_rotmat(hand_pose_aa.reshape(-1, 3))  # (15, 1, 3, 3)
    hand_pose = hand_pose.squeeze(1).unsqueeze(0)  # (1, 15, 3, 3)

    cam_rot = annotation["cam_rot"].astype(np.float32)
    cam_transl = annotation["cam_transl"].astype(np.float32)
    grab2world_R = annotation["grab2world_R"].astype(np.float32)
    grab2world_T = annotation["grab2world_T"].astype(np.float32)

    # 1. MANO forward pass (plain smplx.MANOLayer, outputs MANO-order joints)
    with torch.no_grad():
        mano_out = gt_mano_layer(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas,
        )
    gt_joints_16 = mano_out.joints   # (1, 16, 3) base joints in MANO order
    gt_verts = mano_out.vertices     # (1, 778, 3) meters

    # Add fingertips from mesh vertices (smplx convention, same as HaMeR)
    # smplx: thumb=744, index=320, middle=443, ring=554, pinky=671
    fingertip_vids = [744, 320, 443, 554, 671]
    gt_fingertips = gt_verts[:, fingertip_vids, :]  # (1, 5, 3)
    gt_joints = torch.cat([gt_joints_16, gt_fingertips], dim=1)  # (1, 21, 3) MANO order

    # 2. Coordinate transforms: MANO frame -> camera frame
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

    gt_joints = gt_joints @ all_addition_g_t + all_addition_t_no_transl_t
    gt_joints = gt_joints @ coord_change.T

    gt_verts = gt_verts @ all_addition_g_t + all_addition_t_no_transl_t
    gt_verts = gt_verts @ coord_change.T

    # 3. Root-center (wrist = joint[0])
    root = gt_joints[:, [0], :]
    gt_joints = gt_joints - root
    gt_verts = gt_verts - root

    # Return as numpy, squeezed to (N, 3)
    return gt_joints[0].cpu().numpy(), gt_verts[0].cpu().numpy()


def precompute_and_save_gt(data_dir, mano_dir, device, gt_cache_path):
    """
    Precompute GT 3D joints & vertices for all test samples and save to disk.
    This is model-independent and only needs to be run once.

    Saved format:
        {
            "seq/frame": {
                "joints_3d": (21, 3) float32,   # root-centered, meters
                "verts_3d":  (778, 3) float32,   # root-centered, meters
                "joints_2d": (21, 2) float32,    # pixel coords (x, y)
            },
            ...
        }
    """
    print("=" * 50)
    print("Precomputing GT MANO results (one-time)")
    print("=" * 50)

    gt_mano = create_gt_mano_layer(mano_dir, device)
    samples = load_test_split(data_dir)
    print(f"Test samples: {len(samples)}")

    gt_cache = {}
    skipped = 0

    for seq_name, frame_id, joints_uv in tqdm(samples, desc="Computing GT"):
        key = f"{seq_name}/{frame_id}"
        _, annotation = load_frame_data(data_dir, seq_name, frame_id)
        if annotation is None:
            skipped += 1
            continue

        try:
            gt_j, gt_v = compute_gt_3d(annotation, gt_mano, device)
            gt_cache[key] = {
                "joints_3d": gt_j.astype(np.float32),
                "verts_3d": gt_v.astype(np.float32),
                "joints_2d": joints_uv.astype(np.float32),
            }
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"\n  Skipped {key}: {e}")

    with open(gt_cache_path, "wb") as f:
        pickle.dump(gt_cache, f)

    print(f"\nSaved GT cache: {len(gt_cache)} samples -> {gt_cache_path}")
    if skipped:
        print(f"Skipped: {skipped}")
    return gt_cache


def load_gt_cache(gt_cache_path):
    """Load precomputed GT cache from disk."""
    with open(gt_cache_path, "rb") as f:
        gt_cache = pickle.load(f)
    print(f"Loaded GT cache: {len(gt_cache)} samples from {gt_cache_path}")
    return gt_cache


# ─── HaMeR inference ─────────────────────────────────────────────────────────

def get_bbox_from_joints(joints_uv, expansion_factor=1.5):
    """
    Compute bounding box from 2D joint coordinates.
    Args:
        joints_uv: (21, 2) array of (x, y) pixel coordinates
        expansion_factor: bbox expansion factor
    Returns:
        bbox: (4,) array [x_min, y_min, x_max, y_max]
    """
    x_min, y_min = joints_uv.min(axis=0)
    x_max, y_max = joints_uv.max(axis=0)

    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    w, h = x_max - x_min, y_max - y_min
    size = max(w, h) * expansion_factor

    return np.array([
        cx - size / 2,
        cy - size / 2,
        cx + size / 2,
        cy + size / 2,
    ], dtype=np.float32)


def run_hamer_inference(model, model_cfg, img_cv2, bbox, device):
    """
    Run HaMeR inference on a single image with a given bounding box.
    Returns root-centered pred_joints (1,21,3), pred_verts (1,778,3),
    and pred_2d_pixels (1,21,2) in original image coordinates.
    """
    import io
    from contextlib import redirect_stdout
    from hamer.datasets.vitdet_dataset import ViTDetDataset

    # Create dataset (suppress ViTDetDataset debug prints)
    with redirect_stdout(io.StringIO()):
        dataset = ViTDetDataset(
            model_cfg,
            img_cv2,
            boxes=bbox.reshape(1, 4),
            right=np.array([1.0]),  # POV-Surgery is right-hand only
            rescale_factor=2.5,
        )
        item = dataset[0]

    # Prepare batch
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

    # Forward pass
    with torch.no_grad():
        out = model(batch)

    pred_joints = out["pred_keypoints_3d"]  # (1, 21, 3) OpenPose order
    pred_verts = out["pred_vertices"]        # (1, 778, 3)
    pred_2d = out["pred_keypoints_2d"]       # (1, 21, 2) normalized, OpenPose order

    # Reorder from OpenPose to MANO order to match GT
    pred_joints = pred_joints[:, OPENPOSE_TO_MANO, :]
    pred_2d = pred_2d[:, OPENPOSE_TO_MANO, :]

    # Root-center 3D predictions (wrist = joint[0])
    root_j = pred_joints[:, [0], :]
    pred_joints = pred_joints - root_j
    pred_verts = pred_verts - root_j

    # Map 2D predictions to original image pixel coordinates
    box_center = torch.tensor(item["box_center"], device=device).float()
    box_size = torch.tensor([item["box_size"]], device=device).float()
    pred_2d_pixels = pred_2d[0] * box_size + box_center  # (21, 2)

    return pred_joints, pred_verts, pred_2d_pixels.unsqueeze(0)


# ─── Data loading ────────────────────────────────────────────────────────────

def load_test_split(data_dir):
    """
    Load POV-Surgery test split.
    Returns list of (seq_name, frame_id, joints_uv_xy) tuples.
    """
    test_pkl_path = os.path.join(data_dir, "handoccnet_train",
                                  "2d_repro_ho3d_style_test_cleaned.pkl")
    if not os.path.exists(test_pkl_path):
        raise FileNotFoundError(
            f"Test split pickle not found at {test_pkl_path}\n"
            f"Download POV_Surgery_data from: "
            f"https://drive.google.com/drive/folders/1nSDig2cEHscCPgG10-VcSW3Q1zKge4tP"
        )

    with open(test_pkl_path, "rb") as f:
        test_info = pickle.load(f)

    samples = []
    for key, val in test_info.items():
        seq_name, frame_id = key.split("/")
        # Swap axes: POV-Surgery stores (y, x) -> we need (x, y)
        # From pov_surgery.py lines 326-329
        joints_uv_raw = val["joints_uv"]  # (21, 2)
        joints_uv = np.zeros_like(joints_uv_raw)
        joints_uv[:, 0] = joints_uv_raw[:, 1]  # x = col 1
        joints_uv[:, 1] = joints_uv_raw[:, 0]  # y = col 0
        samples.append((seq_name, frame_id, joints_uv))

    return samples


def load_frame_data(data_dir, seq_name, frame_id):
    """Load image and annotation for a frame."""
    img_path = os.path.join(data_dir, "color", seq_name, f"{frame_id}.jpg")
    anno_path = os.path.join(data_dir, "annotation", seq_name, f"{frame_id}.pkl")

    if not os.path.exists(img_path):
        return None, None
    if not os.path.exists(anno_path):
        return None, None

    img = cv2.imread(img_path)
    with open(anno_path, "rb") as f:
        annotation = pickle.load(f)

    return img, annotation


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate HaMeR on POV-Surgery")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to POV_Surgery_data directory")
    parser.add_argument("--checkpoint", type=str, default="DEFAULT",
                        help="HaMeR checkpoint path (or 'DEFAULT' for pretrained)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples to evaluate (-1 for all)")
    parser.add_argument("--mano_dir", type=str, default=None,
                        help="Path to MANO model directory (default: hamer/_DATA/data/mano)")
    parser.add_argument("--gt_cache", type=str, default=None,
                        help="Path to GT cache file (default: AIM_2_Project/gt_cache_povsurgery_test.pkl)")
    parser.add_argument("--precompute_gt", action="store_true",
                        help="Only precompute and save GT cache, then exit")
    args = parser.parse_args()

    # Convert to absolute paths before chdir
    args.data_dir = os.path.abspath(args.data_dir)
    if args.mano_dir:
        args.mano_dir = os.path.abspath(args.mano_dir)
    if args.checkpoint != "DEFAULT":
        args.checkpoint = os.path.abspath(args.checkpoint)

    gt_cache_path = args.gt_cache or os.path.join(SCRIPT_DIR, GT_CACHE_FILENAME)
    gt_cache_path = os.path.abspath(gt_cache_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── HaMeR uses relative paths, so we must chdir ──
    os.chdir(HAMER_ROOT)
    mano_dir = args.mano_dir or os.path.join(HAMER_ROOT, "_DATA", "data", "mano")

    # ── Precompute GT mode ──
    if args.precompute_gt:
        precompute_and_save_gt(args.data_dir, mano_dir, device, gt_cache_path)
        return

    # ── Load or compute GT cache ──
    if os.path.exists(gt_cache_path):
        gt_cache = load_gt_cache(gt_cache_path)
    else:
        print(f"GT cache not found at {gt_cache_path}")
        print("Computing GT on the fly (run with --precompute_gt to cache for reuse)")
        gt_cache = None

    # ── Load HaMeR model ──
    from hamer.models import load_hamer, download_models, DEFAULT_CHECKPOINT
    from hamer.configs import CACHE_DIR_HAMER

    download_models(CACHE_DIR_HAMER)
    ckpt = DEFAULT_CHECKPOINT if args.checkpoint == "DEFAULT" else args.checkpoint
    print(f"Loading HaMeR from: {ckpt}")
    model, model_cfg = load_hamer(ckpt)
    model = model.to(device)
    model.eval()

    # ── Create GT MANO layer (only if no cache) ──
    gt_mano = None
    if gt_cache is None:
        print(f"Loading GT MANO from: {mano_dir}")
        gt_mano = create_gt_mano_layer(mano_dir, device)

    # ── Load test split ──
    print(f"Loading test split from: {args.data_dir}")
    samples = load_test_split(args.data_dir)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"Total test samples: {len(samples)}")

    # ── Evaluate ──
    metrics_all = {"p_2d": [], "mpjpe": [], "pve": [], "pa_mpjpe": [], "pa_pve": []}
    skipped = 0

    for seq_name, frame_id, joints_uv in tqdm(samples, desc="Evaluating"):
        key = f"{seq_name}/{frame_id}"

        # ── Get GT ──
        if gt_cache is not None and key in gt_cache:
            gt_entry = gt_cache[key]
            gt_joints = torch.tensor(gt_entry["joints_3d"], device=device).unsqueeze(0)
            gt_verts = torch.tensor(gt_entry["verts_3d"], device=device).unsqueeze(0)
            joints_uv = gt_entry["joints_2d"]
        elif gt_mano is not None:
            _, annotation = load_frame_data(args.data_dir, seq_name, frame_id)
            if annotation is None:
                skipped += 1
                continue
            try:
                gt_j_np, gt_v_np = compute_gt_3d(annotation, gt_mano, device)
                gt_joints = torch.tensor(gt_j_np, device=device).unsqueeze(0)
                gt_verts = torch.tensor(gt_v_np, device=device).unsqueeze(0)
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"\n  GT failed {key}: {e}")
                continue
        else:
            skipped += 1
            continue

        # ── Load image ──
        img_path = os.path.join(args.data_dir, "color", seq_name, f"{frame_id}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        try:
            # Compute bbox from GT 2D joints
            bbox = get_bbox_from_joints(joints_uv, expansion_factor=1.5)

            # Run HaMeR
            pred_joints, pred_verts, pred_2d = run_hamer_inference(
                model, model_cfg, img, bbox, device
            )

            # 3D metrics
            m = compute_metrics(pred_joints, gt_joints, pred_verts, gt_verts)
            metrics_all["mpjpe"].append(m["mpjpe"])
            metrics_all["pve"].append(m["pve"])
            metrics_all["pa_mpjpe"].append(m["pa_mpjpe"])
            metrics_all["pa_pve"].append(m["pa_pve"])

            # P_2d (2D joint error in pixels)
            gt_2d = torch.tensor(joints_uv, dtype=torch.float32, device=device)
            p_2d = torch.sqrt(((pred_2d[0] - gt_2d) ** 2).sum(-1)).mean().item()
            metrics_all["p_2d"].append(p_2d)

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"\n  Inference failed {key}: {e}")
            continue

    # ── Report results ──
    n = len(metrics_all["mpjpe"])
    print(f"\n{'=' * 50}")
    print(f"HaMeR Zero-Shot Evaluation on POV-Surgery")
    print(f"{'=' * 50}")
    print(f"Evaluated: {n} samples  |  Skipped: {skipped}")
    print(f"{'=' * 50}")
    print(f"  P_2d:      {np.mean(metrics_all['p_2d']):.2f} pixels")
    print(f"  MPJPE:     {np.mean(metrics_all['mpjpe']):.2f} mm")
    print(f"  PVE:       {np.mean(metrics_all['pve']):.2f} mm")
    print(f"  PA-MPJPE:  {np.mean(metrics_all['pa_mpjpe']):.2f} mm")
    print(f"  PA-PVE:    {np.mean(metrics_all['pa_pve']):.2f} mm")
    print(f"{'=' * 50}")

    print(f"\nPaper Table 2 (finetuned HaMeR):")
    print(f"  P_2d=13.05  MPJPE=13.15  PVE=12.55  PA-MPJPE=4.41  PA-PVE=4.18")

    # Save per-sample results
    results_path = os.path.join(SCRIPT_DIR, "results_hamer.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(metrics_all, f)
    print(f"\nPer-sample results saved to: {results_path}")


if __name__ == "__main__":
    main()
