"""WiLoR FT with multi-teacher ensemble distillation.

Two supervision streams:
  - POV (filtered): true synthetic GT joints_3d → MANO-order L1
  - Aria train: cached MGFM+HONet ensemble pred_3d_op as soft GT → OP-order L1,
                weighted by per-joint teacher-disagreement confidence.

POV out-of-frame samples (joints_2d fully outside image) are filtered at __init__.

Eval: same as ft_wilor_mixed.py, Aria-MPS PA-MPJPE + POV PA-MPJPE.
"""
import os
import sys
import argparse
import json
import io
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
from tqdm import tqdm

sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from eval_joint_orders import MANO_TO_OPENPOSE


POV_DIR = "/workspace/datasets/pov_surgery/data"
ARIA_DIR = "/workspace/datasets/aria_val/data"
TEACHER_CACHE = "/workspace/cache/ensemble_teacher_aria_train.npz"

# OP_VALID: skip index 1 (invalid OP joint)
OP_VALID = [0] + list(range(2, 21))


def _crop_and_normalize(img, joints_2d, hand_side, image_size=256, augment=True, padding=1.5,
                       img_wh=None, joints_3d=None):
    """Returns (img_t (H,W,3 BGR float32), kp3d_with_vis (21,4) or None, flip flag)."""
    if img_wh is None:
        img_wh = (img.shape[1], img.shape[0])
    bbox = derive_bbox_from_joints2d(joints_2d, padding=padding, img_wh=img_wh)
    is_right = 1 if hand_side == "right" else 0

    if augment:
        scale_aug = np.random.uniform(0.85, 1.15)
        shift = np.random.uniform(-0.1, 0.1, 2) * bbox[2:]
        cx, cy = bbox[0] + bbox[2] / 2 + shift[0], bbox[1] + bbox[3] / 2 + shift[1]
        bsize = max(bbox[2], bbox[3]) * scale_aug
    else:
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        bsize = max(bbox[2], bbox[3])

    flip = (is_right == 0)
    if flip:
        img = img[:, ::-1].copy()
        cx = img.shape[1] - cx
        if joints_3d is not None:
            joints_3d = joints_3d.copy()
            joints_3d[:, 0] = -joints_3d[:, 0]

    src = np.array([[cx - bsize / 2, cy - bsize / 2],
                    [cx + bsize / 2, cy - bsize / 2],
                    [cx - bsize / 2, cy + bsize / 2]], dtype=np.float32)
    dst = np.array([[0, 0], [image_size, 0], [0, image_size]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(img, M, (image_size, image_size), flags=cv2.INTER_LINEAR)

    if augment:
        if np.random.random() < 0.5:
            crop = np.clip(crop.astype(np.float32) * np.random.uniform(0.8, 1.2), 0, 255)
        if np.random.random() < 0.3:
            jitter = np.random.uniform(0.9, 1.1, 3)
            crop = np.clip(crop.astype(np.float32) * jitter, 0, 255)

    crop_bgr = crop[:, :, ::-1].copy().astype(np.float32)

    if joints_3d is not None:
        kp3d_root = joints_3d - joints_3d[0:1]
        kp3d_with_vis = np.concatenate([kp3d_root, np.ones((kp3d_root.shape[0], 1), dtype=np.float32)], axis=1)
    else:
        kp3d_with_vis = None

    return crop_bgr.astype(np.float32), kp3d_with_vis, flip


def joints_2d_in_bounds(j2d, img_w, img_h):
    """At least 50% of joints must be within image."""
    in_x = (j2d[:, 0] >= 0) & (j2d[:, 0] < img_w)
    in_y = (j2d[:, 1] >= 0) & (j2d[:, 1] < img_h)
    return (in_x & in_y).mean() > 0.5


class POVTrainDataset(Dataset):
    """POV with out-of-frame filter."""
    def __init__(self, image_size=256, augment=True):
        self.image_size = image_size
        self.augment = augment
        files = sorted(Path(POV_DIR).glob("train-*.parquet"))
        self._meta = []
        kept, dropped = 0, 0
        for f in files:
            tbl = pq.read_table(f, columns=["sequence_name", "frame_id", "joints_2d", "image_width", "image_height"])
            df = tbl.to_pandas()
            for i, r in df.iterrows():
                j2d = np.asarray(r.joints_2d, dtype=np.float32).reshape(21, 2)
                if joints_2d_in_bounds(j2d, int(r.image_width), int(r.image_height)):
                    self._meta.append((str(f), i))
                    kept += 1
                else:
                    dropped += 1
        print(f"  POVTrainDataset: kept {kept}, dropped {dropped} (out-of-frame joints_2d)")
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
        img_t, kp3d, _ = _crop_and_normalize(
            img, joints_2d, r.hand_side,
            image_size=self.image_size, augment=self.augment,
            img_wh=(int(r.image_width), int(r.image_height)),
            joints_3d=joints_3d)
        return {
            "img": img_t,
            "kp3d": kp3d,                     # MANO order (21, 4)
            "kp3d_op": np.zeros((21, 4), dtype=np.float32),  # unused, placeholder
            "confidence": np.zeros((21,), dtype=np.float32),
            "source": 0,                      # 0 = POV, 1 = Aria-distill
        }


def ccw90_2d(j2d, W):
    return np.stack([j2d[:, 1], (W - 1) - j2d[:, 0]], axis=-1).astype(np.float32)


class AriaDistillDataset(Dataset):
    """Aria train w/ ensemble teacher (MGFM+HONet) as soft GT in OP order."""
    def __init__(self, image_size=256, augment=True, apply_reject_filter=True,
                 teacher_path=TEACHER_CACHE):
        self.image_size = image_size
        self.augment = augment

        print(f"  Loading teacher cache from {teacher_path}...")
        c = np.load(teacher_path, allow_pickle=False)
        self.teacher = {str(k): (p, q) for k, p, q in zip(c["keys"], c["pred_3d_op"], c["confidence"])}
        print(f"  teacher entries: {len(self.teacher)}")

        drop = set()
        if apply_reject_filter:
            R = json.load(open("/workspace/datasets/phase0_sidecars/reject_keys_all_20260419.json"))
            drop = set(R.get("train_reject_keys", [])) | set(R.get("train_skip_keys", []))

        files = sorted(Path(ARIA_DIR).glob("train-*.parquet"))
        self._meta = []
        kept, miss_teacher, drop_filter = 0, 0, 0
        for f in files:
            tbl = pq.read_table(f, columns=["sequence_name", "frame_id", "hand_side"])
            df = tbl.to_pandas()
            for i, r in df.iterrows():
                key = f"{r.sequence_name}/{r.frame_id}"
                if key in drop:
                    drop_filter += 1
                    continue
                tkey = f"{r.sequence_name}/{r.frame_id}/{r.hand_side}"
                if tkey not in self.teacher:
                    miss_teacher += 1
                    continue
                self._meta.append((str(f), i, tkey))
                kept += 1
        print(f"  AriaDistillDataset: kept {kept}, dropped {drop_filter} (reject), {miss_teacher} (no teacher)")
        self._dfs = {}

    def __len__(self):
        return len(self._meta)

    def _load(self, fpath, idx):
        if fpath not in self._dfs:
            self._dfs[fpath] = pq.read_table(fpath).to_pandas()
        return self._dfs[fpath].iloc[idx]

    def __getitem__(self, i):
        fpath, idx, tkey = self._meta[i]
        r = self._load(fpath, idx)
        img = np.asarray(Image.open(io.BytesIO(r.image["bytes"])).convert("RGB"))
        W = int(r.image_width)
        joints_2d_raw = np.asarray(r.joints_2d, dtype=np.float32).reshape(21, 2)
        joints_2d = ccw90_2d(joints_2d_raw, W)

        teacher_p3d, teacher_conf = self.teacher[tkey]  # OP order, root-rel
        # Apply flip-handling, teacher is already flipped to canonical right via predict_*
        # _crop_and_normalize will flip image+joints_2d for left hand. We need teacher 3D
        # to match the flipped image space. The teacher cache stored "as if right", so for
        # a left-hand sample we need to also negate teacher x to get back to image-flipped frame.
        # Actually: teacher was computed by flipping the left image to right, predicting, then
        # un-negating x at save time (see precompute_teacher.py: `if not is_right: p3d[:,0] = -p3d[:,0]`).
        # So teacher is in original (left-hand if left) image-space x convention.
        # _crop_and_normalize flips the image for left hands; we must mirror teacher x too.
        is_right = 1 if r.hand_side == "right" else 0
        flip = (is_right == 0)
        teacher_p3d = teacher_p3d.copy()
        if flip:
            teacher_p3d[:, 0] = -teacher_p3d[:, 0]

        img_t, _, _ = _crop_and_normalize(
            img, joints_2d, r.hand_side,
            image_size=self.image_size, augment=self.augment,
            img_wh=(W, int(r.image_height)),
            joints_3d=None)

        kp3d_op = np.concatenate([teacher_p3d, np.ones((21, 1), dtype=np.float32)], axis=1).astype(np.float32)
        return {
            "img": img_t,
            "kp3d": np.zeros((21, 4), dtype=np.float32),  # placeholder
            "kp3d_op": kp3d_op,                           # (21, 4) OP order, vis=1
            "confidence": teacher_conf.astype(np.float32),  # (21,)
            "source": 1,                                  # 1 = Aria-distill
        }


_MANO_TO_OP_T = torch.tensor(MANO_TO_OPENPOSE, dtype=torch.long)
_OP_VALID_T = torch.tensor(OP_VALID, dtype=torch.long)


def distill_loss(pred_mano, kp3d, kp3d_op, confidence, source, root_idx=0):
    """source=0: MANO L1 vs kp3d.   source=1: convert pred_mano→OP, L1 vs kp3d_op (conf-weighted, OP_VALID)."""
    device = pred_mano.device
    mano_to_op = _MANO_TO_OP_T.to(device)
    op_valid = _OP_VALID_T.to(device)

    pred_rel = pred_mano - pred_mano[:, root_idx:root_idx+1, :]

    # POV (source==0): MANO order L1
    gt_mano = kp3d[..., :3]
    gt_mano_rel = gt_mano - gt_mano[:, root_idx:root_idx+1, :]
    pov_diff = (pred_rel - gt_mano_rel).abs().mean(dim=(1, 2))   # (B,)

    # Aria (source==1): convert pred to OP, L1 vs teacher in OP_VALID
    pred_op = pred_rel.index_select(dim=1, index=mano_to_op)     # (B, 21, 3)
    pred_op_rel = pred_op - pred_op[:, root_idx:root_idx+1, :]
    gt_op = kp3d_op[..., :3]
    gt_op_rel = gt_op - gt_op[:, root_idx:root_idx+1, :]
    diff_op = (pred_op_rel - gt_op_rel).abs().mean(dim=-1)        # (B, 21)
    diff_op_v = diff_op.index_select(dim=1, index=op_valid)       # (B, 20)
    # Uniform weighting: per-teacher disagreement is dominated by frame disagreement,
    # not joint reliability, confidence-weighting biases gradient away from useful joints.
    aria_diff = diff_op_v.mean(dim=1)                              # (B,)

    src_mask = source.float()
    loss_per = (1.0 - src_mask) * pov_diff + src_mask * aria_diff
    return loss_per.mean(), pov_diff.mean(), aria_diff.mean()


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
        meta.append({"row": r, "flip": flip})

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
    agg, _ = run_metrics_aggregation(samples, dataset_name, "WiLoR-FT-distill", True)
    model.train()
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="/workspace/checkpoints/wilor_ft_distill")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_every_steps", type=int, default=500)
    p.add_argument("--aria_weight", type=float, default=1.0)
    p.add_argument("--no_aug", action="store_true")
    p.add_argument("--no_pov", action="store_true")
    p.add_argument("--no_aria", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/5] Loading WiLoR pretrained...")
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    pipe = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
    model = pipe.wilor_model
    model.train()
    n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"  trainable: {n_train/1e6:.1f}M")

    print("[2/5] Loading datasets...")
    if args.no_aria and not args.no_pov:
        pov_ds = POVTrainDataset(image_size=256, augment=not args.no_aug)
        mixed_ds = pov_ds
        weights = [1.0] * len(pov_ds)
    elif args.no_pov and not args.no_aria:
        aria_ds = AriaDistillDataset(image_size=256, augment=not args.no_aug)
        mixed_ds = aria_ds
        weights = [1.0] * len(aria_ds)
    else:
        pov_ds = POVTrainDataset(image_size=256, augment=not args.no_aug)
        aria_ds = AriaDistillDataset(image_size=256, augment=not args.no_aug)
        mixed_ds = ConcatDataset([pov_ds, aria_ds])
        weights = ([1.0] * len(pov_ds)) + ([args.aria_weight * len(pov_ds) / len(aria_ds)] * len(aria_ds))
        print(f"  Mixed n={len(mixed_ds)} (POV={len(pov_ds)}, Aria-distill={len(aria_ds)})")
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(mixed_ds), replacement=True)
    train_loader = DataLoader(mixed_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=(args.num_workers > 0))

    print("[3/5] Loading eval sets...")
    pov_eval = load_pov_test(stride=1)
    aria_eval = load_aria_val()
    print(f"  POV eval n={len(pov_eval)}, Aria eval n={len(aria_eval)}")

    optim = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad],
                              lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()

    print("[4/5] Baseline eval...")
    pov_agg = evaluate(model, pov_eval, "pov_test")
    aria_agg = evaluate(model, aria_eval, "aria_val")
    pov0 = pov_agg.get("pov_native_pa_mpjpe_mm", float("nan"))
    aria0 = aria_agg.get("aria_native_pa_mpjpe_mm", float("nan"))
    hsam0 = aria_agg.get("hsam_pa_mpjpe_mm", float("nan"))
    print(f"  baseline POV: {pov0:.2f}  Aria-MPS: {aria0:.2f}  HSAM: {hsam0:.2f}")
    with open(f"{args.out_dir}/eval_log.jsonl", "a") as f:
        f.write(json.dumps({"step": 0, "pov_pa_mpjpe": pov0, "aria_pa_mpjpe": aria0, "hsam_pa_mpjpe": hsam0}) + "\n")

    print("[5/5] Training...")
    step = 0
    best_aria = aria0
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", dynamic_ncols=True)
        for batch in pbar:
            imgs = batch["img"].cuda(non_blocking=True)
            kp3d = batch["kp3d"].cuda(non_blocking=True)
            kp3d_op = batch["kp3d_op"].cuda(non_blocking=True)
            conf = batch["confidence"].cuda(non_blocking=True)
            src = batch["source"].cuda(non_blocking=True)

            with autocast(dtype=torch.bfloat16):
                out = model(imgs)
                pred_3d = out["pred_keypoints_3d"]
                loss, pov_l, aria_l = distill_loss(pred_3d, kp3d, kp3d_op, conf, src)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_([p_ for p_ in model.parameters() if p_.requires_grad], 1.0)
            scaler.step(optim)
            scaler.update()
            pbar.set_postfix(loss=f"{float(loss):.4f}", pov=f"{float(pov_l):.3f}", aria=f"{float(aria_l):.3f}")
            step += 1

            if step % args.eval_every_steps == 0:
                print(f"\n[step {step}] eval...")
                pov_agg = evaluate(model, pov_eval, "pov_test")
                aria_agg = evaluate(model, aria_eval, "aria_val")
                pov_pa = pov_agg.get("pov_native_pa_mpjpe_mm", float("nan"))
                aria_pa = aria_agg.get("aria_native_pa_mpjpe_mm", float("nan"))
                hsam_pa = aria_agg.get("hsam_pa_mpjpe_mm", float("nan"))
                print(f"  POV: {pov_pa:.2f}  Aria-MPS: {aria_pa:.2f}  HSAM: {hsam_pa:.2f}")

                torch.save({"step": step, "model_state_dict": model.state_dict(),
                            "pov_pa_mpjpe": pov_pa, "aria_pa_mpjpe": aria_pa},
                           f"{args.out_dir}/wilor_ft_step{step}.pth")
                if aria_pa < best_aria:
                    best_aria = aria_pa
                    torch.save({"step": step, "model_state_dict": model.state_dict(),
                                "pov_pa_mpjpe": pov_pa, "aria_pa_mpjpe": aria_pa},
                               f"{args.out_dir}/wilor_ft_best_aria.pth")
                    print(f"  ⭐ new best Aria-MPS: {aria_pa:.2f}")
                with open(f"{args.out_dir}/eval_log.jsonl", "a") as f:
                    f.write(json.dumps({"step": step, "pov_pa_mpjpe": pov_pa,
                                        "aria_pa_mpjpe": aria_pa, "hsam_pa_mpjpe": hsam_pa}) + "\n")

    print(f"\n[done]  best Aria-MPS: {best_aria:.2f} mm")
    pov_agg = evaluate(model, pov_eval, "pov_test")
    aria_agg = evaluate(model, aria_eval, "aria_val")
    print(f"FINAL, POV: {pov_agg.get('pov_native_pa_mpjpe_mm', float('nan')):.2f}  "
          f"Aria-MPS: {aria_agg.get('aria_native_pa_mpjpe_mm', float('nan')):.2f}")
    torch.save({"model_state_dict": model.state_dict()}, f"{args.out_dir}/wilor_ft_final.pth")


if __name__ == "__main__":
    main()
