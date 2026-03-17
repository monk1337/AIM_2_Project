"""
Visualize model predictions in 256x256 crop space.
Both HaMeR and HandOCCNet predict on 256x256 crops — this script
shows GT vs pred 2D joints overlaid on the crop for direct comparison.

For each sample, generates a 1x2 panel:
  [Left]  Crop with GT 2D joints (green) and Pred 2D joints (red)
  [Right] Same crop with skeleton overlay

Usage:
  python visualize_crop.py --data_dir /path/to/POV_Surgery_data --model hamer --n_samples 10
  CUDA_VISIBLE_DEVICES=0 python visualize_crop.py --data_dir /path/to/POV_Surgery_data --model handoccnet --n_samples 10
"""

import os
import sys
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')

SCRIPT_DIR = str(Path(__file__).resolve().parent)
HAMER_ROOT = str(Path(__file__).resolve().parent.parent / "hamer")
HANDOCC_ROOT = str(Path(__file__).resolve().parent.parent / "POV_Surgery" / "HandOccNet_ft")

# MANO-order skeleton
HAND_EDGES_MANO = [
    (0, 13), (13, 14), (14, 15), (15, 16),  # thumb
    (0, 1), (1, 2), (2, 3), (3, 17),        # index
    (0, 4), (4, 5), (5, 6), (6, 18),        # middle
    (0, 10), (10, 11), (11, 12), (12, 19),  # ring
    (0, 7), (7, 8), (8, 9), (9, 20),        # pinky
]
EDGE_FINGER = ['thumb'] * 4 + ['index'] * 4 + ['middle'] * 4 + ['ring'] * 4 + ['pinky'] * 4

# OpenPose-order skeleton (for HandOCCNet which reorders to OpenPose internally)
HAND_EDGES_OPENPOSE = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
]

COLORS = {
    'thumb':  (255, 100, 100),
    'index':  (100, 255, 100),
    'middle': (100, 100, 255),
    'ring':   (255, 255, 100),
    'pinky':  (255, 100, 255),
}


def draw_skeleton(img, joints, edges, edge_finger, color_override=None, thickness=2, radius=3):
    """Draw skeleton on image. joints: (21, 2) pixel coords."""
    vis = img.copy()
    h, w = vis.shape[:2]
    pts = joints.astype(int)

    for i, (p, c) in enumerate(edges):
        col = color_override or COLORS[edge_finger[i]]
        pt1, pt2 = tuple(pts[p]), tuple(pts[c])
        if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
            0 <= pt2[0] < w and 0 <= pt2[1] < h):
            cv2.line(vis, pt1, pt2, col, thickness, cv2.LINE_AA)

    for i, (x, y) in enumerate(pts):
        if 0 <= x < w and 0 <= y < h:
            col = color_override or COLORS[edge_finger[min(i, len(edge_finger)-1)]]
            cv2.circle(vis, (x, y), radius, col, -1, cv2.LINE_AA)
            cv2.circle(vis, (x, y), radius, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def unnormalize_imagenet(img_tensor):
    """Convert ImageNet-normalized tensor (3,H,W) back to uint8 BGR image."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3) RGB
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ─── HaMeR ───────────────────────────────────────────────────────────────────

def get_hamer_crops(model, model_cfg, data_dir, samples, device, n):
    """Get HaMeR crop images, GT 2D, and pred 2D in 256x256 space."""
    import io
    from contextlib import redirect_stdout
    sys.path.insert(0, SCRIPT_DIR)
    from evaluate_hamer import get_bbox_from_joints, OPENPOSE_TO_MANO
    from hamer.datasets.vitdet_dataset import ViTDetDataset

    results = []
    for seq, fid, juv in samples[:n]:
        img_path = os.path.join(data_dir, "color", seq, f"{fid}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            continue

        bbox = get_bbox_from_joints(juv, expansion_factor=1.5)
        with redirect_stdout(io.StringIO()):
            ds = ViTDetDataset(model_cfg, img, boxes=bbox.reshape(1, 4),
                               right=np.array([1.0]), rescale_factor=2.5)
            item = ds[0]

        # Get crop image (undo ImageNet normalization)
        crop_img = unnormalize_imagenet(torch.tensor(item['img']))

        # Run inference
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

        # Pred 2D: centered in [-0.5, 0.5] -> pixel [0, 256]
        pred_2d = out['pred_keypoints_2d'][0].cpu().numpy()  # (21, 2) OpenPose order
        pred_2d_px = (pred_2d + 0.5) * 256  # to pixel coords
        pred_2d_px = pred_2d_px[OPENPOSE_TO_MANO]  # to MANO order

        # GT 2D: project GT joints_uv into crop space via the same affine
        # We approximate by using box_center/box_size
        box_center = item['box_center']  # (2,)
        box_size = item['box_size']      # scalar
        gt_2d_px = (juv - box_center) / box_size * 256 + 128  # approximate mapping
        # Note: this is approximate. For exact mapping we'd need the affine transform.

        results.append({
            'key': f"{seq}/{fid}",
            'crop': crop_img,
            'gt_2d': gt_2d_px,      # (21, 2) MANO order
            'pred_2d': pred_2d_px,   # (21, 2) MANO order
            'joint_order': 'mano',
        })

    return results


# ─── HandOCCNet ──────────────────────────────────────────────────────────────

def get_handoccnet_crops(model, data_dir, device, n):
    """Get HandOCCNet crop images, GT 2D, and pred 2D in 256x256 space."""
    from torchvision import transforms
    from torch.utils.data import DataLoader

    sys.path.insert(0, os.path.join(HANDOCC_ROOT, 'data'))
    sys.path.insert(0, os.path.join(HANDOCC_ROOT, 'data', 'pov_surgery'))
    from pov_surgery import POVSURGERY

    dataset = POVSURGERY(transforms.ToTensor(), 'validation')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # OpenPose joint order mapping (used inside model.py line 192)
    jointsMapManoToSimple = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

    results = []
    for itr, (inputs, targets, meta_info) in enumerate(loader):
        if len(results) >= n:
            break

        inp_img = inputs['img']  # (1, 3, 256, 256)
        crop_img = (inp_img[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)

        inputs_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        targets_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

        step = str(itr * 2 + 1).zfill(6)
        with torch.no_grad():
            out = model(inputs_cuda, targets_cuda, meta_info, 'my_val', this_name=step)

        # Extract pred and GT 2D joints from model internals
        # The model reorders GT to OpenPose (line 192) and recovers to [0,1] crop coords (lines 194-203)
        # pred_joints_img is also [0,1] in crop
        # We need to hook into the model to get these... Instead, let's reconstruct:

        # GT 2D: targets['joints2d'] was already reordered to OpenPose and normalized to bbox
        # In the model, it gets recover_joints then /256 -> [0,1] in crop
        # We can reconstruct from original targets
        bbox_hand = targets['bbox_hand'][0].numpy()  # (4,) [x1, y1, x2, y2]
        joints2d_norm = targets['joints2d'][0].numpy()  # (21, 2) normalized to bbox, MANO order
        # Reorder to OpenPose (same as model line 192)
        joints2d_openpose = joints2d_norm[jointsMapManoToSimple]
        # Recover to crop pixel coords
        gt_2d_px = joints2d_openpose * (bbox_hand[2:] - bbox_hand[:2]) + bbox_hand[:2]

        # Run full model forward (which does the float conversion etc.)
        # The model's forward computes pred 2D joints internally as preds_joints_img
        # but doesn't expose them. We'll approximate pred 2D from the metric:
        # Since model returns 'joints_img' (2D error), we can also get pred by
        # running the regressor after the model has prepared targets.

        # Simpler: reconstruct pred 2D from the model's internal state.
        # We hook into the regressor to capture preds_joints_img.
        captured = {}
        orig_forward = model.module.regressor.forward

        def hooked_forward(feats, gt_mano_params=None):
            res = orig_forward(feats, gt_mano_params)
            captured['preds_joints_img'] = res[2]
            return res

        model.module.regressor.forward = hooked_forward
        step = str(itr * 2 + 1).zfill(6)
        with torch.no_grad():
            out = model(inputs_cuda, targets_cuda, meta_info, 'my_val', this_name=step)
        model.module.regressor.forward = orig_forward

        pred_2d_norm = captured['preds_joints_img'][0][0].detach().cpu().numpy()  # (21, 2) [0,1] OpenPose
        pred_2d_px = pred_2d_norm * (bbox_hand[2:] - bbox_hand[:2]) + bbox_hand[:2]

        seq = targets['seqName'][0] if 'seqName' in targets else f"sample_{itr}"
        fid = targets['id'][0] if 'id' in targets else str(itr)

        results.append({
            'key': f"{seq}/{fid}",
            'crop': crop_img,
            'gt_2d': gt_2d_px,       # (21, 2) OpenPose order
            'pred_2d': pred_2d_px,    # (21, 2) OpenPose order
            'joint_order': 'openpose',
        })

    return results


# ─── Visualization ───────────────────────────────────────────────────────────

def create_panel(crop, gt_2d, pred_2d, joint_order, key, model_name):
    """Create a side-by-side panel: GT overlay | Pred overlay."""
    edges = HAND_EDGES_MANO if joint_order == 'mano' else HAND_EDGES_OPENPOSE

    # Left: GT (green)
    left = draw_skeleton(crop, gt_2d, edges, EDGE_FINGER, color_override=(0, 220, 0))
    cv2.putText(left, "GT", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

    # Right: Pred (red)
    right = draw_skeleton(crop, pred_2d, edges, EDGE_FINGER, color_override=(0, 0, 220))
    cv2.putText(right, "Pred", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)

    # Center: both overlaid
    center = draw_skeleton(crop, gt_2d, edges, EDGE_FINGER, color_override=(0, 220, 0), thickness=1, radius=2)
    center = draw_skeleton(center, pred_2d, edges, EDGE_FINGER, color_override=(0, 0, 220), thickness=1, radius=2)
    cv2.putText(center, "Both", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    panel = np.concatenate([left, center, right], axis=1)

    # Title bar
    title_h = 25
    title = np.zeros((title_h, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(title, f"{model_name} | {key}", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.concatenate([title, panel], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["hamer", "handoccnet"])
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hamer_ckpt", type=str, default="DEFAULT")
    parser.add_argument("--handoccnet_ckpt", type=str,
                        default=os.path.join(SCRIPT_DIR, "checkpoints",
                                             "HandOccNet_model_dump", "HO3D", "snapshot_80.pth.tar"))
    args = parser.parse_args()

    args.data_dir = os.path.abspath(args.data_dir)
    output_dir = args.output_dir or os.path.join(SCRIPT_DIR, "visualizations_crop")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "HaMeR" if args.model == "hamer" else "HandOCCNet"

    if args.model == "hamer":
        sys.path.insert(0, HAMER_ROOT)
        os.chdir(HAMER_ROOT)
        from hamer.models import load_hamer, download_models, DEFAULT_CHECKPOINT
        from hamer.configs import CACHE_DIR_HAMER
        download_models(CACHE_DIR_HAMER)
        ckpt = DEFAULT_CHECKPOINT if args.hamer_ckpt == "DEFAULT" else args.hamer_ckpt
        model, model_cfg = load_hamer(ckpt)
        model = model.to(device).eval()

        sys.path.insert(0, SCRIPT_DIR)
        from evaluate_hamer import load_test_split
        samples = load_test_split(args.data_dir)

        np.random.seed(args.seed)
        idxs = np.random.choice(len(samples), min(args.n_samples, len(samples)), replace=False)
        selected = [samples[i] for i in idxs]

        print(f"Running {model_name} on {len(selected)} samples...")
        results = get_hamer_crops(model, model_cfg, args.data_dir, selected, device, len(selected))

    elif args.model == "handoccnet":
        import torch.nn as nn
        sys.path.insert(0, HANDOCC_ROOT)
        sys.path.insert(0, os.path.join(HANDOCC_ROOT, 'main'))
        sys.path.insert(0, os.path.join(HANDOCC_ROOT, 'common'))
        from nets.backbone import FPN
        from nets.transformer import Transformer
        from nets.regressor import Regressor
        from main.model import Model

        backbone = FPN(pretrained=False)
        fIT = Transformer(injection=True)
        sET = Transformer(injection=False)
        regressor = Regressor()
        model = Model(backbone, fIT, sET, regressor)
        model = nn.DataParallel(model).cuda()
        ckpt = torch.load(args.handoccnet_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        # Select random samples by iterating
        np.random.seed(args.seed)
        print(f"Running {model_name} on {args.n_samples} samples...")
        results = get_handoccnet_crops(model, args.data_dir, device, args.n_samples)

    # Save panels
    for i, r in enumerate(results):
        panel = create_panel(r['crop'], r['gt_2d'], r['pred_2d'],
                             r['joint_order'], r['key'], model_name)
        safe_key = r['key'].replace('/', '_')
        out_path = os.path.join(output_dir, f"{args.model}_{safe_key}.jpg")
        cv2.imwrite(out_path, panel)
        print(f"  [{i+1}/{len(results)}] {r['key']} -> {out_path}")

    print(f"\nDone! Saved to {output_dir}")


if __name__ == "__main__":
    main()
