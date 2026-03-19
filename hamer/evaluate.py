"""
Unified evaluation script for hand mesh recovery models on POV-Surgery.

Supports: HaMeR, HandOCCNet (zero-shot or finetuned).

Computes the same 5 metrics as Table 2 in the MICCAI paper:
  P_2d     - 2D joint reprojection error (pixels)
  MPJPE    - Mean Per-Joint Position Error (mm)
  PVE      - Per-Vertex Error (mm)
  PA-MPJPE - Procrustes-Aligned MPJPE (mm)
  PA-PVE   - Procrustes-Aligned PVE (mm)

Usage:
  # Step 1: Precompute GT (one-time, shared across models)
  python evaluate.py --data_dir /path/to/POV_Surgery_data --precompute_gt

  # Step 2: Evaluate a model
  python evaluate.py --data_dir /path/to/POV_Surgery_data --model hamer
  python evaluate.py --data_dir /path/to/POV_Surgery_data --model handoccnet \
      --handoccnet_ckpt /path/to/snapshot_80.pth.tar
"""

import os
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = str(Path(__file__).resolve().parent)
HAMER_ROOT = str(Path(__file__).resolve().parent.parent / "hamer")
HANDOCC_ROOT = str(Path(__file__).resolve().parent.parent / "POV_Surgery" / "HandOccNet_ft")

GT_CACHE_FILENAME = "gt_cache_povsurgery_test.pkl"


# ─── Shared GT and Metrics (from evaluate_hamer.py) ─────────────────────────

def load_gt_cache(gt_cache_path):
    with open(gt_cache_path, "rb") as f:
        gt_cache = pickle.load(f)
    print(f"Loaded GT cache: {len(gt_cache)} samples from {gt_cache_path}")
    return gt_cache


def precompute_gt(data_dir, gt_cache_path, device):
    """Precompute GT using the pipeline from evaluate_hamer.py."""
    sys.path.insert(0, HAMER_ROOT)
    os.chdir(HAMER_ROOT)
    sys.path.insert(0, SCRIPT_DIR)
    from evaluate_hamer import precompute_and_save_gt
    mano_dir = os.path.join(HAMER_ROOT, "_DATA", "data", "mano")
    precompute_and_save_gt(data_dir, mano_dir, device, gt_cache_path)


def load_test_split(data_dir):
    """Load POV-Surgery test split. Returns list of (seq, frame, joints_uv_xy)."""
    test_pkl = os.path.join(data_dir, "handoccnet_train",
                            "2d_repro_ho3d_style_test_cleaned.pkl")
    with open(test_pkl, "rb") as f:
        test_info = pickle.load(f)
    samples = []
    for key, val in test_info.items():
        seq, fid = key.split("/")
        juv_raw = val["joints_uv"]
        juv = np.zeros_like(juv_raw)
        juv[:, 0] = juv_raw[:, 1]
        juv[:, 1] = juv_raw[:, 0]
        samples.append((seq, fid, juv))
    return samples


# ─── HaMeR Backend ───────────────────────────────────────────────────────────

def setup_hamer(args, device):
    """Load HaMeR model. Returns (model, model_cfg)."""
    sys.path.insert(0, HAMER_ROOT)
    os.chdir(HAMER_ROOT)
    from hamer.models import load_hamer, download_models, DEFAULT_CHECKPOINT
    from hamer.configs import CACHE_DIR_HAMER
    download_models(CACHE_DIR_HAMER)
    ckpt = DEFAULT_CHECKPOINT if args.hamer_ckpt == "DEFAULT" else args.hamer_ckpt
    print(f"Loading HaMeR from: {ckpt}")
    model, model_cfg = load_hamer(ckpt)
    model = model.to(device).eval()
    return {"model": model, "model_cfg": model_cfg}


def run_hamer_sample(ctx, img_path, joints_uv, gt_joints, gt_verts, device):
    """
    Run HaMeR on one sample. Returns dict of metrics or None on failure.
    """
    import cv2
    import io
    from contextlib import redirect_stdout
    sys.path.insert(0, SCRIPT_DIR)
    from evaluate_hamer import get_bbox_from_joints, compute_metrics, OPENPOSE_TO_MANO
    from hamer.datasets.vitdet_dataset import ViTDetDataset

    img = cv2.imread(img_path)
    if img is None:
        return None

    bbox = get_bbox_from_joints(joints_uv, expansion_factor=1.5)
    model = ctx["model"]
    model_cfg = ctx["model_cfg"]

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

    pred_joints = out["pred_keypoints_3d"][:, OPENPOSE_TO_MANO, :]
    pred_verts = out["pred_vertices"]
    pred_2d = out["pred_keypoints_2d"][:, OPENPOSE_TO_MANO, :]

    root_j = pred_joints[:, [0], :]
    pred_joints = pred_joints - root_j
    pred_verts = pred_verts - root_j

    box_center = torch.tensor(item["box_center"], device=device).float()
    box_size = torch.tensor([item["box_size"]], device=device).float()
    pred_2d_pixels = pred_2d[0] * box_size + box_center

    m = compute_metrics(pred_joints, gt_joints, pred_verts, gt_verts)

    gt_2d = torch.tensor(joints_uv, dtype=torch.float32, device=device)
    p_2d = torch.sqrt(((pred_2d_pixels - gt_2d) ** 2).sum(-1)).mean().item()
    m["p_2d"] = p_2d
    return m


# ─── HandOCCNet Backend ──────────────────────────────────────────────────────

def setup_handoccnet(args, device):
    """Load HandOCCNet model. Returns context dict."""
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

    ckpt_path = args.handoccnet_ckpt
    print(f"Loading HandOCCNet from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['network'], strict=False)
    model.eval()
    print(f"HandOCCNet loaded (epoch {ckpt.get('epoch', '?')})")
    return {"model": model}


def run_handoccnet_eval(ctx, data_dir, samples, gt_cache, device, max_samples):
    """
    Run HandOCCNet evaluation using its native data loader.

    HandOCCNet computes GT and metrics internally in its forward pass
    (using manopth MANO), which ensures consistency with the paper's Table 2.
    """
    from torchvision import transforms
    from torch.utils.data import DataLoader

    sys.path.insert(0, os.path.join(HANDOCC_ROOT, 'data'))
    sys.path.insert(0, os.path.join(HANDOCC_ROOT, 'data', 'pov_surgery'))
    from pov_surgery import POVSURGERY

    dataset = POVSURGERY(transforms.ToTensor(), 'validation')
    if max_samples > 0:
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = ctx["model"]
    metrics_all = {"p_2d": [], "mpjpe": [], "pve": [], "pa_mpjpe": [], "pa_pve": []}
    skipped = 0

    for itr, (inputs, targets, meta_info) in enumerate(tqdm(loader, desc="Evaluating HandOCCNet")):
        try:
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            targets = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

            # Use step_this not divisible by 10 to skip debug viz code
            step = str(itr * 2 + 1).zfill(6)
            with torch.no_grad():
                out = model(inputs, targets, meta_info, 'my_val', this_name=step)

            # HandOCCNet returns per-sample metrics directly
            metrics_all["p_2d"].append(out['joints_img'].item())
            metrics_all["mpjpe"].append(out['mano_joints'].item())
            metrics_all["pve"].append(out['mano_verts'].item())
            metrics_all["pa_mpjpe"].append(out['j3d_pa'].item())
            metrics_all["pa_pve"].append(out['v3d_pa'].item())
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"\n  Failed sample {itr}: {e}")

    return metrics_all, skipped


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate hand mesh recovery on POV-Surgery")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to POV_Surgery_data directory")
    parser.add_argument("--model", type=str, default="hamer",
                        choices=["hamer", "handoccnet"],
                        help="Model to evaluate")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--gt_cache", type=str, default=None)
    parser.add_argument("--precompute_gt", action="store_true")

    # HaMeR-specific
    parser.add_argument("--hamer_ckpt", type=str, default="DEFAULT")

    # HandOCCNet-specific
    parser.add_argument("--handoccnet_ckpt", type=str,
                        default=os.path.join(SCRIPT_DIR, "checkpoints",
                                             "HandOccNet_model_dump", "HO3D", "snapshot_80.pth.tar"))
    args = parser.parse_args()

    args.data_dir = os.path.abspath(args.data_dir)
    gt_cache_path = args.gt_cache or os.path.join(SCRIPT_DIR, GT_CACHE_FILENAME)
    gt_cache_path = os.path.abspath(gt_cache_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Precompute GT ──
    if args.precompute_gt:
        precompute_gt(args.data_dir, gt_cache_path, device)
        return

    # ── Evaluate ──
    model_name = args.model.upper()
    if model_name == "HAMER":
        model_name = "HaMeR"

    if args.model == "hamer":
        # HaMeR uses our shared GT cache + per-sample inference
        os.chdir(HAMER_ROOT)
        sys.path.insert(0, HAMER_ROOT)

        if os.path.exists(gt_cache_path):
            gt_cache = load_gt_cache(gt_cache_path)
        else:
            print(f"ERROR: GT cache not found at {gt_cache_path}")
            print("Run with --precompute_gt first.")
            return

        ctx = setup_hamer(args, device)
        samples = load_test_split(args.data_dir)
        if args.max_samples > 0:
            samples = samples[:args.max_samples]
        print(f"Total test samples: {len(samples)}")

        metrics_all = {"p_2d": [], "mpjpe": [], "pve": [], "pa_mpjpe": [], "pa_pve": []}
        skipped = 0

        for seq, fid, juv in tqdm(samples, desc=f"Evaluating {model_name}"):
            key = f"{seq}/{fid}"
            if key not in gt_cache:
                skipped += 1
                continue

            gt_entry = gt_cache[key]
            gt_joints = torch.tensor(gt_entry["joints_3d"], device=device).unsqueeze(0)
            gt_verts = torch.tensor(gt_entry["verts_3d"], device=device).unsqueeze(0)
            juv = gt_entry["joints_2d"]

            img_path = os.path.join(args.data_dir, "color", seq, f"{fid}.jpg")
            try:
                m = run_hamer_sample(ctx, img_path, juv, gt_joints, gt_verts, device)
                if m is None:
                    skipped += 1
                    continue
                for k in metrics_all:
                    metrics_all[k].append(m[k])
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"\n  Failed {key}: {e}")

    elif args.model == "handoccnet":
        # HandOCCNet uses its own data loader + internal GT/metrics
        gt_cache = None  # not used
        ctx = setup_handoccnet(args, device)
        metrics_all, skipped = run_handoccnet_eval(
            ctx, args.data_dir, None, gt_cache, device, args.max_samples)

    # ── Report ──
    n = len(metrics_all["mpjpe"])
    print(f"\n{'=' * 55}")
    print(f"  {model_name} Zero-Shot on POV-Surgery")
    print(f"{'=' * 55}")
    print(f"  Evaluated: {n} samples  |  Skipped: {skipped}")
    print(f"{'=' * 55}")
    print(f"  P_2d:      {np.mean(metrics_all['p_2d']):>8.2f} pixels")
    print(f"  MPJPE:     {np.mean(metrics_all['mpjpe']):>8.2f} mm")
    print(f"  PVE:       {np.mean(metrics_all['pve']):>8.2f} mm")
    print(f"  PA-MPJPE:  {np.mean(metrics_all['pa_mpjpe']):>8.2f} mm")
    print(f"  PA-PVE:    {np.mean(metrics_all['pa_pve']):>8.2f} mm")
    print(f"{'=' * 55}")
    print(f"\n  Paper Table 2 (finetuned):")
    print(f"  {'Method':<15} {'P_2d':>6} {'MPJPE':>7} {'PVE':>7} {'PA-MPJPE':>9} {'PA-PVE':>7}")
    print(f"  {'HaMeR':<15} {'13.05':>6} {'13.15':>7} {'12.55':>7} {'4.41':>9} {'4.18':>7}")
    print(f"  {'HandOCCNet':<15} {'13.80':>6} {'14.35':>7} {'13.73':>7} {'4.49':>9} {'4.35':>7}")
    print(f"  {'CPCI (best)':<15} {'12.08':>6} {'12.21':>7} {'12.25':>7} {'4.21':>9} {'4.20':>7}")

    # Save results
    results_path = os.path.join(SCRIPT_DIR, f"results_{args.model}.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(metrics_all, f)
    print(f"\nPer-sample results saved to: {results_path}")


if __name__ == "__main__":
    main()
