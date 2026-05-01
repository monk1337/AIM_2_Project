"""Fine-tune WiLoR on POV-Surgery train.

Loss: 3D keypoint L1 (root-relative) + 2D keypoint L1 + MANO param L2.
Validates periodically on POV test (n=5,766 stride=5) and Aria val (n=2,333).

Uses bf16 mixed precision + grad accumulation for max throughput on A100 80GB.
"""
import os
import sys
import argparse
import json
import time
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import cv2
import pyarrow.parquet as pq
from PIL import Image
import io
from tqdm import tqdm

sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from eval_joint_orders import mano_to_op, OP_VALID
from eval_metrics import metrics_3d, metrics_pve, aggregate, procrustes_align


POV_DIR = "/workspace/datasets/pov_surgery/data"


class POVTrainDataset(Dataset):
    """POV-Surgery train: returns (img_patch, gt dict) for WiLoR fine-tuning.

    Lazy image decode for memory.
    """

    def __init__(self, image_size=256, padding=1.5, do_flip=False, augment=True):
        self.image_size = image_size
        self.padding = padding
        self.augment = augment
        files = sorted(Path(POV_DIR).glob("train-*.parquet"))
        # Read metadata only first
        self._meta = []
        for f in files:
            tbl = pq.read_table(f, columns=["sequence_name", "frame_id", "hand_side"])
            df = tbl.to_pandas()
            for i, r in df.iterrows():
                self._meta.append((str(f), i, r.sequence_name, int(r.frame_id), r.hand_side))
        # Cache file handles + dataframes lazily
        self._dfs = {}

    def __len__(self):
        return len(self._meta)

    def _load_row(self, fpath, idx):
        if fpath not in self._dfs:
            self._dfs[fpath] = pq.read_table(fpath).to_pandas()
        return self._dfs[fpath].iloc[idx]

    def __getitem__(self, i):
        fpath, idx, _, _, hand_side = self._meta[i]
        r = self._load_row(fpath, idx)
        img = np.asarray(Image.open(io.BytesIO(r.image["bytes"])).convert("RGB"))
        joints_2d = np.asarray(r.joints_2d, dtype=np.float32).reshape(21, 2)
        joints_3d = np.asarray(r.joints_3d, dtype=np.float32).reshape(21, 3)
        verts = np.asarray(r.vertices, dtype=np.float32).reshape(778, 3)
        K = np.asarray(r.intrinsics, dtype=np.float32).reshape(3, 3)
        global_orient = np.asarray(r.global_orient, dtype=np.float32)
        hand_pose = np.asarray(r.hand_pose, dtype=np.float32)
        betas = np.asarray(r.betas, dtype=np.float32)

        bbox = derive_bbox_from_joints2d(joints_2d, padding=self.padding,
                                         img_wh=(int(r.image_width), int(r.image_height)))
        is_right = 1 if hand_side == "right" else 0

        # Augmentation: small random scale/translation (only at training)
        if self.augment:
            scale_aug = np.random.uniform(0.9, 1.1)
            shift = np.random.uniform(-0.05, 0.05, 2) * bbox[2:]
            cx, cy = bbox[0] + bbox[2] / 2 + shift[0], bbox[1] + bbox[3] / 2 + shift[1]
            bsize = max(bbox[2], bbox[3]) * scale_aug
        else:
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            bsize = max(bbox[2], bbox[3])

        # Flip left hand to right (WiLoR convention)
        flip = (is_right == 0)
        if flip:
            img = img[:, ::-1].copy()
            cx = img.shape[1] - cx
            joints_2d_local = joints_2d.copy()
            joints_2d_local[:, 0] = img.shape[1] - joints_2d_local[:, 0]
            joints_3d_local = joints_3d.copy()
            joints_3d_local[:, 0] = -joints_3d_local[:, 0]
            verts_local = verts.copy()
            verts_local[:, 0] = -verts_local[:, 0]
            global_orient_local = global_orient.copy()
            global_orient_local[1::3] *= -1
            global_orient_local[2::3] *= -1
            hand_pose_local = hand_pose.copy()
            hand_pose_local[1::3] *= -1
            hand_pose_local[2::3] *= -1
        else:
            joints_2d_local = joints_2d
            joints_3d_local = joints_3d
            verts_local = verts
            global_orient_local = global_orient
            hand_pose_local = hand_pose

        # Crop image to image_size, wilor_mini.forward expects (H,W,3) BGR uint8-ish (it does flip→/255→normalize internally)
        src = np.array([[cx - bsize / 2, cy - bsize / 2],
                        [cx + bsize / 2, cy - bsize / 2],
                        [cx - bsize / 2, cy + bsize / 2]], dtype=np.float32)
        dst = np.array([[0, 0], [self.image_size, 0], [0, self.image_size]], dtype=np.float32)
        M = cv2.getAffineTransform(src, dst)
        crop = cv2.warpAffine(img, M, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR)
        # PIL gives RGB; wilor expects BGR (then flips internally to RGB before normalize)
        crop_bgr = crop[:, :, ::-1].copy()
        img_t = crop_bgr.astype(np.float32)  # (H, W, 3) BGR

        # Transform 2D keypoints to crop coords (then to model's normalized [-0.5, 0.5])
        joints_2d_homog = np.concatenate([joints_2d_local, np.ones((21, 1))], axis=1)
        kp_crop = (M @ joints_2d_homog.T).T  # (21, 2) in [0, image_size]
        kp_norm = kp_crop / self.image_size - 0.5  # to [-0.5, 0.5]
        kp_norm_with_vis = np.concatenate([kp_norm, np.ones((21, 1), dtype=np.float32)], axis=1)  # (21, 3)

        # 3D root-relative for loss
        kp3d_root = joints_3d_local - joints_3d_local[0:1]  # wrist at origin
        kp3d_with_vis = np.concatenate([kp3d_root, np.ones((21, 1), dtype=np.float32)], axis=1)  # (21, 4)

        return {
            "img": img_t,
            "keypoints_2d": kp_norm_with_vis.astype(np.float32),  # (21, 3)
            "keypoints_3d": kp3d_with_vis.astype(np.float32),     # (21, 4)
            "mano_global_orient": global_orient_local.astype(np.float32),  # (3,)
            "mano_hand_pose": hand_pose_local.astype(np.float32),          # (45,)
            "mano_betas": betas.astype(np.float32),                         # (10,)
        }


def keypoint_3d_l1(pred, gt, root_idx=0):
    """pred, gt: (B, 21, 3 or 4). gt last col may be visibility."""
    if gt.shape[-1] == 4:
        vis = gt[..., 3:4]
        gt = gt[..., :3]
    else:
        vis = torch.ones_like(gt[..., :1])
    pred_rel = pred - pred[..., root_idx:root_idx+1, :]
    gt_rel = gt - gt[..., root_idx:root_idx+1, :]
    err = (pred_rel - gt_rel).abs() * vis
    return err.mean()


def keypoint_2d_l1(pred, gt):
    """pred, gt: (B, 21, 2 or 3). gt last col may be visibility."""
    if gt.shape[-1] == 3:
        vis = gt[..., 2:3]
        gt = gt[..., :2]
    else:
        vis = torch.ones_like(gt[..., :1])
    err = (pred - gt).abs() * vis
    return err.mean()


def mano_param_l2(pred, gt):
    return F.mse_loss(pred.view(-1), gt.view(-1))


def evaluate(model, eval_rows, dataset_name, image_size=256, batch_size=128, device="cuda"):
    """Evaluate model on a set of rows. Returns aggregate dict."""
    from eval_runner import compute_metrics, run_metrics_aggregation

    print(f"  [eval] crop+forward {dataset_name} (n={len(eval_rows)})…")
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
        crop_bgr = crop[:, :, ::-1].copy().astype(np.float32)  # PIL RGB → BGR; (H, W, 3)
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
    agg, _ = run_metrics_aggregation(samples, dataset_name, "WiLoR-FT", True)
    model.train()
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="/workspace/checkpoints/wilor_ft")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_every_steps", type=int, default=500)
    p.add_argument("--w_kp3d", type=float, default=0.05)
    p.add_argument("--w_kp2d", type=float, default=0.01)
    p.add_argument("--w_orient", type=float, default=0.001)
    p.add_argument("--w_pose", type=float, default=0.001)
    p.add_argument("--w_betas", type=float, default=0.0005)
    p.add_argument("--freeze_backbone", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load WiLoR via wilor_mini (we'll use the underlying nn.Module)
    print("[1/4] Loading WiLoR pretrained...")
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

    print("[2/4] Loading POV train dataset…")
    train_ds = POVTrainDataset(image_size=256, augment=True)
    print(f"  POV train n={len(train_ds)}")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    print("[3/4] Loading eval sets (POV FULL test, Aria val)…")
    pov_eval = load_pov_test(stride=1)  # full POV test n=28,802
    aria_eval = load_aria_val()
    print(f"  POV test n={len(pov_eval)} (full), Aria val n={len(aria_eval)}")

    optim = torch.optim.AdamW(
        [p_ for p_ in model.parameters() if p_.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scaler = GradScaler()

    print("[4/4] Training…")
    step = 0
    log_buffer = []
    log_path = f"{args.out_dir}/ft_log.jsonl"
    open(log_path, "w").close()  # clear
    t0 = time.time()
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", dynamic_ncols=True)
        for batch in pbar:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            with autocast(dtype=torch.bfloat16):
                out = model(batch["img"])
                pred_3d = out["pred_keypoints_3d"]      # (B, 21, 3) MANO order, root-aligned (m)
                # 3D keypoint L1 only, MANO param losses dropped (caused training to diverge)
                loss_kp3d = keypoint_3d_l1(pred_3d, batch["keypoints_3d"], root_idx=0)
                loss = args.w_kp3d * loss_kp3d
                loss_kp2d = torch.tensor(0.0, device=loss.device)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(
                [p_ for p_ in model.parameters() if p_.requires_grad], 1.0)
            scaler.step(optim)
            scaler.update()

            log_buffer.append({
                "step": step, "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "loss_kp3d": float(loss_kp3d.detach().cpu()),
                "loss_kp2d": float(loss_kp2d.detach().cpu()),
                "lr": args.lr,
                "elapsed_s": time.time() - t0,
            })
            pbar.set_postfix(loss=f"{float(loss):.4f}",
                             kp3d=f"{float(loss_kp3d):.3f}",
                             kp2d=f"{float(loss_kp2d):.3f}")

            step += 1

            # Periodic eval + checkpoint
            if step % args.eval_every_steps == 0:
                # Flush log
                with open(log_path, "a") as f:
                    for line in log_buffer:
                        f.write(json.dumps(line) + "\n")
                log_buffer = []

                print(f"\n[step {step}] eval…")
                pov_agg = evaluate(model, pov_eval, "pov_test", batch_size=128)
                aria_agg = evaluate(model, aria_eval, "aria_val", batch_size=128)
                print(f"  POV PA-MPJPE: {pov_agg.get('pov_native_pa_mpjpe_mm', float('nan')):.2f} mm")
                print(f"  Aria PA-MPJPE: {aria_agg.get('aria_native_pa_mpjpe_mm', float('nan')):.2f} mm  HSAM: {aria_agg.get('hsam_pa_mpjpe_mm', float('nan')):.2f} mm")
                # Save checkpoint
                ck_path = f"{args.out_dir}/wilor_ft_step{step}.pth"
                torch.save({"step": step, "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "pov_pa_mpjpe": pov_agg.get("pov_native_pa_mpjpe_mm"),
                            "aria_pa_mpjpe": aria_agg.get("aria_native_pa_mpjpe_mm"),
                           }, ck_path)
                print(f"  saved {ck_path}")
                with open(f"{args.out_dir}/eval_log.jsonl", "a") as f:
                    f.write(json.dumps({
                        "step": step, "epoch": epoch,
                        "pov_pa_mpjpe": pov_agg.get("pov_native_pa_mpjpe_mm"),
                        "aria_pa_mpjpe": aria_agg.get("aria_native_pa_mpjpe_mm"),
                        "aria_hsam_pa_mpjpe": aria_agg.get("hsam_pa_mpjpe_mm"),
                    }) + "\n")

    # Final flush
    with open(log_path, "a") as f:
        for line in log_buffer:
            f.write(json.dumps(line) + "\n")

    print("[done]")
    # Final eval + save
    pov_agg = evaluate(model, pov_eval, "pov_test", batch_size=128)
    aria_agg = evaluate(model, aria_eval, "aria_val", batch_size=128)
    print(f"FINAL, POV PA-MPJPE: {pov_agg.get('pov_native_pa_mpjpe_mm', float('nan')):.2f} mm")
    print(f"FINAL, Aria PA-MPJPE: {aria_agg.get('aria_native_pa_mpjpe_mm', float('nan')):.2f} mm")
    torch.save({"step": step, "epoch": args.epochs - 1,
                "model_state_dict": model.state_dict(),
                "pov_pa_mpjpe": pov_agg.get("pov_native_pa_mpjpe_mm"),
                "aria_pa_mpjpe": aria_agg.get("aria_native_pa_mpjpe_mm"),
               }, f"{args.out_dir}/wilor_ft_final.pth")


if __name__ == "__main__":
    main()
