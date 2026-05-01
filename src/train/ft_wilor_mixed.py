"""WiLoR FT on POV + Aria train (HSAM) mixed.

Aria train has HSAM pseudo-labels in MANO order (root-relative-ish, learned from HaMeR).
POV train has true synthetic MANO GT.
Both use kp3d L1 loss in root-relative frame.

Aria val (PR81-85) is held-out and never in training.
"""
import os
import sys
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
import cv2
import pyarrow.parquet as pq
from PIL import Image
import io
from tqdm import tqdm

sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from eval_joint_orders import mano_to_op


POV_DIR = "/workspace/datasets/pov_surgery/data"
ARIA_DIR = "/workspace/datasets/aria_val/data"


def _crop_and_normalize(img, joints_2d, joints_3d, hand_side, image_size=256, augment=True, padding=1.5,
                       img_wh=None):
    """Shared crop+aug logic. Returns (img_t (H,W,3 BGR float32), kp_norm_with_vis (21,3), kp3d_with_vis (21,4))."""
    if img_wh is None:
        img_wh = (img.shape[1], img.shape[0])
    bbox = derive_bbox_from_joints2d(joints_2d, padding=padding, img_wh=img_wh)
    is_right = 1 if hand_side == "right" else 0

    # Augmentation
    if augment:
        scale_aug = np.random.uniform(0.85, 1.15)
        shift = np.random.uniform(-0.1, 0.1, 2) * bbox[2:]
        cx, cy = bbox[0] + bbox[2] / 2 + shift[0], bbox[1] + bbox[3] / 2 + shift[1]
        bsize = max(bbox[2], bbox[3]) * scale_aug
    else:
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        bsize = max(bbox[2], bbox[3])

    # Flip left → right
    flip = (is_right == 0)
    if flip:
        img = img[:, ::-1].copy()
        cx = img.shape[1] - cx
        joints_2d_local = joints_2d.copy()
        joints_2d_local[:, 0] = img.shape[1] - joints_2d_local[:, 0]
        joints_3d_local = joints_3d.copy()
        joints_3d_local[:, 0] = -joints_3d_local[:, 0]
    else:
        joints_2d_local = joints_2d
        joints_3d_local = joints_3d

    # Crop
    src = np.array([[cx - bsize / 2, cy - bsize / 2],
                    [cx + bsize / 2, cy - bsize / 2],
                    [cx - bsize / 2, cy + bsize / 2]], dtype=np.float32)
    dst = np.array([[0, 0], [image_size, 0], [0, image_size]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(img, M, (image_size, image_size), flags=cv2.INTER_LINEAR)

    # Color augmentation (training only)
    if augment:
        # Brightness
        if np.random.random() < 0.5:
            crop = np.clip(crop.astype(np.float32) * np.random.uniform(0.8, 1.2), 0, 255)
        # Channel jitter
        if np.random.random() < 0.3:
            jitter = np.random.uniform(0.9, 1.1, 3)
            crop = np.clip(crop.astype(np.float32) * jitter, 0, 255)

    # PIL is RGB; wilor expects BGR (then flips internally to RGB before normalize)
    crop_bgr = crop[:, :, ::-1].copy().astype(np.float32)
    img_t = crop_bgr  # (H, W, 3) BGR

    # 2D in crop coords
    joints_2d_homog = np.concatenate([joints_2d_local, np.ones((21, 1))], axis=1)
    kp_crop = (M @ joints_2d_homog.T).T
    kp_norm = kp_crop / image_size - 0.5
    kp_norm_with_vis = np.concatenate([kp_norm, np.ones((21, 1), dtype=np.float32)], axis=1)

    # 3D root-relative
    kp3d_root = joints_3d_local - joints_3d_local[0:1]
    kp3d_with_vis = np.concatenate([kp3d_root, np.ones((21, 1), dtype=np.float32)], axis=1)

    return img_t.astype(np.float32), kp_norm_with_vis.astype(np.float32), kp3d_with_vis.astype(np.float32)


class POVTrainDataset(Dataset):
    def __init__(self, image_size=256, augment=True):
        self.image_size = image_size
        self.augment = augment
        files = sorted(Path(POV_DIR).glob("train-*.parquet"))
        self._meta = []
        for f in files:
            tbl = pq.read_table(f, columns=["sequence_name", "frame_id", "hand_side"])
            df = tbl.to_pandas()
            for i, _ in df.iterrows():
                self._meta.append((str(f), i))
        self._dfs = {}

    def __len__(self):
        return len(self._meta)

    def _load(self, fpath, idx):
        if fpath not in self._dfs:
            self._dfs[fpath] = pq.read_table(fpath).to_pandas()
        return self._dfs[fpath].iloc[idx]

    def __getitem__(self, i):
        fpath, idx = self._meta[i]
        r = self._load(fpath, idx)
        img = np.asarray(Image.open(io.BytesIO(r.image["bytes"])).convert("RGB"))
        joints_2d = np.asarray(r.joints_2d, dtype=np.float32).reshape(21, 2)
        joints_3d = np.asarray(r.joints_3d, dtype=np.float32).reshape(21, 3)
        img_t, kp2d, kp3d = _crop_and_normalize(
            img, joints_2d, joints_3d, r.hand_side,
            image_size=self.image_size, augment=self.augment,
            img_wh=(int(r.image_width), int(r.image_height)))
        return {
            "img": img_t, "keypoints_2d": kp2d, "keypoints_3d": kp3d,
            "source": "pov",
        }


def ccw90_2d(j2d, W):
    return np.stack([j2d[:, 1], (W - 1) - j2d[:, 0]], axis=-1).astype(np.float32)


class AriaTrainDataset(Dataset):
    """Aria train HSAM pseudo-labels (joints_3d in MANO local, joints_2d after CCW90).

    Applies the published train_reject_keys filter (sidecar JSON, see SETUP.md)
    to mirror v2-clean cleaning of the val set.
    """
    def __init__(self, image_size=256, augment=True, apply_reject_filter=True):
        self.image_size = image_size
        self.augment = augment
        # Load reject keys
        drop = set()
        if apply_reject_filter:
            import json as _json
            R = _json.load(open("/workspace/datasets/phase0_sidecars/reject_keys_all_20260419.json"))
            drop = set(R.get("train_reject_keys", [])) | set(R.get("train_skip_keys", []))
            print(f"  AriaTrain: applying reject filter ({len(drop)} keys)")
        files = sorted(Path(ARIA_DIR).glob("train-*.parquet"))
        self._meta = []
        kept, skipped = 0, 0
        for f in files:
            tbl = pq.read_table(f, columns=["sequence_name", "frame_id", "hand_side", "is_gt"])
            df = tbl.to_pandas()
            for i, r in df.iterrows():
                key = f"{r.sequence_name}/{r.frame_id}"
                if key in drop:
                    skipped += 1
                    continue
                self._meta.append((str(f), i))
                kept += 1
        if apply_reject_filter:
            print(f"  AriaTrain: kept {kept}, skipped {skipped} via reject filter")
        self._dfs = {}

    def __len__(self):
        return len(self._meta)

    def _load(self, fpath, idx):
        if fpath not in self._dfs:
            self._dfs[fpath] = pq.read_table(fpath).to_pandas()
        return self._dfs[fpath].iloc[idx]

    def __getitem__(self, i):
        fpath, idx = self._meta[i]
        r = self._load(fpath, idx)
        img = np.asarray(Image.open(io.BytesIO(r.image["bytes"])).convert("RGB"))
        W = int(r.image_width)
        joints_2d_raw = np.asarray(r.joints_2d, dtype=np.float32).reshape(21, 2)
        joints_2d = ccw90_2d(joints_2d_raw, W)  # CCW90 to display frame
        joints_3d = np.asarray(r.joints_3d, dtype=np.float32).reshape(21, 3)  # MANO order
        img_t, kp2d, kp3d = _crop_and_normalize(
            img, joints_2d, joints_3d, r.hand_side,
            image_size=self.image_size, augment=self.augment,
            img_wh=(W, int(r.image_height)))
        return {
            "img": img_t, "keypoints_2d": kp2d, "keypoints_3d": kp3d,
            "source": "aria_hsam",
        }


def keypoint_3d_l1(pred, gt, root_idx=0):
    if gt.shape[-1] == 4:
        vis = gt[..., 3:4]
        gt = gt[..., :3]
    else:
        vis = torch.ones_like(gt[..., :1])
    pred_rel = pred - pred[..., root_idx:root_idx+1, :]
    gt_rel = gt - gt[..., root_idx:root_idx+1, :]
    return ((pred_rel - gt_rel).abs() * vis).mean()


def evaluate(model, eval_rows, dataset_name, image_size=256, batch_size=128, device="cuda"):
    from eval_runner import compute_metrics, run_metrics_aggregation
    print(f"  [eval] {dataset_name} (n={len(eval_rows)})…")
    model.eval()
    samples = []
    patches, meta = [], []
    for r in eval_rows:
        img = np.asarray(get_image(r))
        bbox = derive_bbox_from_joints2d(r["native_joints_2d"], padding=1.5, img_wh=r["image_wh"])
        is_right = 1 if r["hand_side"] == "right" else 0
        x_, y_, w_, h_ = bbox
        cx, cy = x_ + w_ / 2, y_ + h_ / 2
        bsize = max(w_, h_)
        flip = (is_right == 0)
        if flip:
            img = img[:, ::-1].copy()
            cx = img.shape[1] - cx
        src = np.array([[cx - bsize / 2, cy - bsize / 2],
                        [cx + bsize / 2, cy - bsize / 2],
                        [cx - bsize / 2, cy + bsize / 2]], dtype=np.float32)
        dst = np.array([[0, 0], [image_size, 0], [0, image_size]], dtype=np.float32)
        M = cv2.getAffineTransform(src, dst)
        crop = cv2.warpAffine(img, M, (image_size, image_size), flags=cv2.INTER_LINEAR)
        crop_bgr = crop[:, :, ::-1].copy().astype(np.float32)
        patches.append(crop_bgr)
        meta.append({"row": r, "flip": flip, "is_right": is_right, "bbox": bbox, "img_wh": r["image_wh"]})

    bs = batch_size
    for i in range(0, len(patches), bs):
        x = torch.from_numpy(np.stack(patches[i:i+bs])).to(device, dtype=torch.float32)
        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            out = model(x)
        for j, m in enumerate(meta[i:i+bs]):
            r = m["row"]
            pred_3d = out["pred_keypoints_3d"][j].float().cpu().numpy()
            pred_verts = out["pred_vertices"][j].float().cpu().numpy()
            if m["flip"]:
                pred_3d = pred_3d.copy(); pred_3d[:, 0] = -pred_3d[:, 0]
                pred_verts = pred_verts.copy(); pred_verts[:, 0] = -pred_verts[:, 0]
            samples.append({
                "row": r,
                "pred_3d_mano": pred_3d,
                "pred_2d_mano": np.zeros((21, 2), dtype=np.float32),
                "pred_verts_mano": pred_verts,
                "pred_cam_t_full": np.zeros(3, dtype=np.float32),
            })
    samples = compute_metrics(samples, dataset_name, root_aligned_model=True)
    agg, _ = run_metrics_aggregation(samples, dataset_name, "WiLoR-FT-mixed", True)
    model.train()
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="/workspace/checkpoints/wilor_ft_mixed")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_every_steps", type=int, default=1000)
    p.add_argument("--w_kp3d", type=float, default=1.0)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--aria_weight", type=float, default=1.0, help="Sampling weight for Aria vs POV")
    p.add_argument("--no_aug", action="store_true")
    p.add_argument("--no_pov", action="store_true", help="Aria-only training")
    p.add_argument("--no_aria", action="store_true", help="POV-only training")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/5] Loading WiLoR pretrained...")
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    pipe = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
    model = pipe.wilor_model
    model.train()

    if args.freeze_backbone:
        for n, p_ in model.named_parameters():
            if n.startswith("backbone"):
                p_.requires_grad = False
        print("  Backbone frozen.")

    n_train_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"  trainable params: {n_train_params/1e6:.1f}M")

    print("[2/5] Loading datasets…")
    if args.no_aria:
        pov_ds = POVTrainDataset(image_size=256, augment=not args.no_aug)
        print(f"  POV train n={len(pov_ds)} (Aria DISABLED)")
        mixed_ds = pov_ds
        weights = [1.0] * len(pov_ds)
    elif args.no_pov:
        aria_ds = AriaTrainDataset(image_size=256, augment=not args.no_aug)
        print(f"  Aria train n={len(aria_ds)} (POV DISABLED)")
        mixed_ds = aria_ds
        weights = [1.0] * len(aria_ds)
    else:
        pov_ds = POVTrainDataset(image_size=256, augment=not args.no_aug)
        aria_ds = AriaTrainDataset(image_size=256, augment=not args.no_aug)
        print(f"  POV train n={len(pov_ds)}")
        print(f"  Aria train n={len(aria_ds)}")
        mixed_ds = ConcatDataset([pov_ds, aria_ds])
        print(f"  Mixed n={len(mixed_ds)}")
        weights = ([1.0] * len(pov_ds)) + ([args.aria_weight * len(pov_ds) / len(aria_ds)] * len(aria_ds))
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(mixed_ds), replacement=True)
    train_loader = DataLoader(
        mixed_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    print("[3/5] Loading eval sets…")
    pov_eval = load_pov_test(stride=1)  # full POV test
    aria_eval = load_aria_val()
    print(f"  POV test n={len(pov_eval)}, Aria val n={len(aria_eval)}")

    optim = torch.optim.AdamW(
        [p_ for p_ in model.parameters() if p_.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scaler = GradScaler()

    print("[4/5] Initial eval (pretrained baseline)...")
    pov_agg = evaluate(model, pov_eval, "pov_test", batch_size=128)
    aria_agg = evaluate(model, aria_eval, "aria_val", batch_size=128)
    print(f"  baseline POV: {pov_agg.get('pov_native_pa_mpjpe_mm', float('nan')):.2f} mm  Aria: {aria_agg.get('aria_native_pa_mpjpe_mm', float('nan')):.2f} mm")
    with open(f"{args.out_dir}/eval_log.jsonl", "a") as f:
        f.write(json.dumps({"step": 0, "epoch": -1,
            "pov_pa_mpjpe": pov_agg.get("pov_native_pa_mpjpe_mm"),
            "aria_pa_mpjpe": aria_agg.get("aria_native_pa_mpjpe_mm"),
            "aria_hsam_pa_mpjpe": aria_agg.get("hsam_pa_mpjpe_mm")}) + "\n")

    print("[5/5] Training…")
    step = 0
    best_aria = aria_agg.get("aria_native_pa_mpjpe_mm", float("inf"))
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", dynamic_ncols=True)
        for batch in pbar:
            imgs = batch["img"].cuda(non_blocking=True)
            kp3d = batch["keypoints_3d"].cuda(non_blocking=True)
            with autocast(dtype=torch.bfloat16):
                out = model(imgs)
                pred_3d = out["pred_keypoints_3d"]
                loss_kp3d = keypoint_3d_l1(pred_3d, kp3d, root_idx=0)
                loss = args.w_kp3d * loss_kp3d
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(
                [p_ for p_ in model.parameters() if p_.requires_grad], 1.0)
            scaler.step(optim)
            scaler.update()
            pbar.set_postfix(loss=f"{float(loss):.4f}", kp3d=f"{float(loss_kp3d):.3f}")
            step += 1

            if step % args.eval_every_steps == 0:
                print(f"\n[step {step}] eval…")
                pov_agg = evaluate(model, pov_eval, "pov_test", batch_size=128)
                aria_agg = evaluate(model, aria_eval, "aria_val", batch_size=128)
                pov_pa = pov_agg.get("pov_native_pa_mpjpe_mm", float("nan"))
                aria_pa = aria_agg.get("aria_native_pa_mpjpe_mm", float("nan"))
                hsam_pa = aria_agg.get("hsam_pa_mpjpe_mm", float("nan"))
                print(f"  POV: {pov_pa:.2f}  Aria: {aria_pa:.2f}  HSAM: {hsam_pa:.2f}")

                ck_path = f"{args.out_dir}/wilor_ft_step{step}.pth"
                torch.save({"step": step, "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "pov_pa_mpjpe": pov_pa,
                            "aria_pa_mpjpe": aria_pa,
                           }, ck_path)
                # Save best-Aria checkpoint separately
                if aria_pa < best_aria:
                    best_aria = aria_pa
                    torch.save({"step": step, "epoch": epoch, "model_state_dict": model.state_dict(),
                                "pov_pa_mpjpe": pov_pa, "aria_pa_mpjpe": aria_pa,
                               }, f"{args.out_dir}/wilor_ft_best_aria.pth")
                    print(f"  ⭐ new best Aria: {aria_pa:.2f}")
                with open(f"{args.out_dir}/eval_log.jsonl", "a") as f:
                    f.write(json.dumps({
                        "step": step, "epoch": epoch,
                        "pov_pa_mpjpe": pov_pa,
                        "aria_pa_mpjpe": aria_pa,
                        "aria_hsam_pa_mpjpe": hsam_pa,
                    }) + "\n")

    print(f"\n[done]  best Aria: {best_aria:.2f} mm")
    pov_agg = evaluate(model, pov_eval, "pov_test", batch_size=128)
    aria_agg = evaluate(model, aria_eval, "aria_val", batch_size=128)
    print(f"FINAL, POV: {pov_agg.get('pov_native_pa_mpjpe_mm', float('nan')):.2f} mm  Aria: {aria_agg.get('aria_native_pa_mpjpe_mm', float('nan')):.2f} mm")
    torch.save({"step": step, "epoch": args.epochs - 1,
                "model_state_dict": model.state_dict()},
               f"{args.out_dir}/wilor_ft_final.pth")


if __name__ == "__main__":
    main()
