"""Fine-tune HandOccNet on POV + Aria train (HSAM) mixed.

HandOccNet outputs joints in OP order, root-aligned, meters.
Loss: 3D L1 in MANO order (we convert OP back to MANO via inv_perm before loss).

Held out: Aria val v2-clean (PR81-85, n=2333).
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

# Wire HandOccNet's import paths, local manopth must come BEFORE pip's
HONET = "/workspace/code/HandOccNet"
sys.path.insert(0, f"{HONET}/common/utils/manopth")
sys.path.insert(0, f"{HONET}/main")
sys.path.insert(0, f"{HONET}/common")

sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from eval_joint_orders import mano_to_op


POV_DIR = "/workspace/datasets/pov_surgery/data"
ARIA_DIR = "/workspace/datasets/aria_val/data"

# OP→MANO inverse permutation (since HandOccNet outputs OP order)
MANO_TO_OP = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20], dtype=np.int64)
OP_TO_MANO = np.argsort(MANO_TO_OP)


def _crop(img, joints_2d, hand_side, image_size=256, augment=True, padding=1.5, img_wh=None):
    """Returns (img_t (C,H,W) [0,1] float, kp3d_local_for_loss_3d_root)."""
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

    src = np.array([[cx - bsize / 2, cy - bsize / 2],
                    [cx + bsize / 2, cy - bsize / 2],
                    [cx - bsize / 2, cy + bsize / 2]], dtype=np.float32)
    dst = np.array([[0, 0], [image_size, 0], [0, image_size]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(img, M, (image_size, image_size), flags=cv2.INTER_LINEAR)

    if augment:
        if np.random.random() < 0.5:
            crop = np.clip(crop.astype(np.float32) * np.random.uniform(0.8, 1.2), 0, 255)

    img_t = (crop.astype(np.float32) / 255.0).transpose(2, 0, 1)  # (3, H, W) RGB [0, 1]
    return img_t, flip


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
        img_t, flip = _crop(img, joints_2d, r.hand_side, image_size=self.image_size,
                            augment=self.augment, img_wh=(int(r.image_width), int(r.image_height)))
        # Apply flip to 3D x
        if flip:
            joints_3d = joints_3d.copy()
            joints_3d[:, 0] = -joints_3d[:, 0]
        # Convert MANO → OP order (HandOccNet outputs OP)
        joints_3d_op = joints_3d[MANO_TO_OP]
        joints_3d_op_root = joints_3d_op - joints_3d_op[0:1]
        return {"img": img_t.astype(np.float32),
                "kp3d_op": joints_3d_op_root.astype(np.float32)}


def ccw90_2d(j2d, W):
    return np.stack([j2d[:, 1], (W - 1) - j2d[:, 0]], axis=-1).astype(np.float32)


class AriaTrainDataset(Dataset):
    def __init__(self, image_size=256, augment=True):
        self.image_size = image_size
        self.augment = augment
        files = sorted(Path(ARIA_DIR).glob("train-*.parquet"))
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
        W = int(r.image_width)
        joints_2d_raw = np.asarray(r.joints_2d, dtype=np.float32).reshape(21, 2)
        joints_2d = ccw90_2d(joints_2d_raw, W)
        joints_3d = np.asarray(r.joints_3d, dtype=np.float32).reshape(21, 3)
        img_t, flip = _crop(img, joints_2d, r.hand_side, image_size=self.image_size,
                            augment=self.augment, img_wh=(W, int(r.image_height)))
        if flip:
            joints_3d = joints_3d.copy()
            joints_3d[:, 0] = -joints_3d[:, 0]
        joints_3d_op = joints_3d[MANO_TO_OP]
        joints_3d_op_root = joints_3d_op - joints_3d_op[0:1]
        return {"img": img_t.astype(np.float32),
                "kp3d_op": joints_3d_op_root.astype(np.float32)}


def keypoint_3d_l1(pred, gt, root_idx=0):
    pred_rel = pred - pred[..., root_idx:root_idx+1, :]
    gt_rel = gt - gt[..., root_idx:root_idx+1, :]
    return (pred_rel - gt_rel).abs().mean()


def evaluate(model, eval_rows, dataset_name, image_size=256, batch_size=64):
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
        img_t = (crop.astype(np.float32) / 255.0).transpose(2, 0, 1)
        patches.append(img_t)
        meta.append({"row": r, "flip": flip, "is_right": is_right, "bbox": bbox, "img_wh": r["image_wh"]})

    bs = batch_size
    for i in range(0, len(patches), bs):
        x = torch.from_numpy(np.stack(patches[i:i+bs])).to("cuda", dtype=torch.float32)
        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            out = model({"img": x}, {}, {}, "test")
        joints_op = out["joints_coord_cam"].float().cpu().numpy()  # (B, 21, 3) OP
        verts = out["mesh_coord_cam"].float().cpu().numpy()
        for j, m in enumerate(meta[i:i+bs]):
            r = m["row"]
            p3d = joints_op[j].copy()
            pverts = verts[j].copy()
            if m["flip"]:
                p3d[:, 0] = -p3d[:, 0]
                pverts[:, 0] = -pverts[:, 0]
            samples.append({
                "row": r,
                "pred_3d_mano": p3d,  # actually OP order
                "pred_2d_mano": np.zeros((21, 2), dtype=np.float32),
                "pred_verts_mano": pverts,
                "pred_cam_t_full": np.zeros(3, dtype=np.float32),
            })
    samples = compute_metrics(samples, dataset_name, root_aligned_model=True, pred_in_op_order=True)
    agg, _ = run_metrics_aggregation(samples, dataset_name, "HandOccNet-FT", True)
    model.train()
    return agg


def load_honet_model():
    from config import cfg as honet_cfg
    honet_cfg.set_args("0")
    target_dir = "/workspace/checkpoints/handoccnet/_mano_root/mano/models"
    os.makedirs(target_dir, exist_ok=True)
    for src_name, name in [("MANO_RIGHT.pkl", "MANO_RIGHT.pkl"), ("MANO_LEFT.pkl", "MANO_LEFT.pkl")]:
        src = f"/workspace/mano/{src_name}"
        dst = f"{target_dir}/{name}"
        if not os.path.exists(dst):
            os.symlink(src, dst)
    honet_cfg.mano_path = "/workspace/checkpoints/handoccnet/_mano_root"

    from model import get_model as honet_get_model
    from torch.nn.parallel.data_parallel import DataParallel
    model = honet_get_model("test")
    model = DataParallel(model).cuda()
    ckpt = torch.load("/workspace/checkpoints/handoccnet/HandOccNet_model_dump/snapshot_demo.pth.tar",
                      map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["network"], strict=False)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="/workspace/checkpoints/honet_ft_mixed")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_every_steps", type=int, default=1000)
    p.add_argument("--aria_weight", type=float, default=1.0)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--no_aug", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/5] Loading HandOccNet pretrained...")
    model = load_honet_model()
    model.train()

    if args.freeze_backbone:
        for n, p_ in model.named_parameters():
            if "backbone" in n:
                p_.requires_grad = False
        print("  Backbone frozen.")

    n_train_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"  trainable params: {n_train_params/1e6:.1f}M")

    print("[2/5] Loading datasets…")
    pov_ds = POVTrainDataset(image_size=256, augment=not args.no_aug)
    aria_ds = AriaTrainDataset(image_size=256, augment=not args.no_aug)
    print(f"  POV n={len(pov_ds)}  Aria n={len(aria_ds)}")
    mixed_ds = ConcatDataset([pov_ds, aria_ds])
    weights = ([1.0] * len(pov_ds)) + ([args.aria_weight * len(pov_ds) / len(aria_ds)] * len(aria_ds))
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(mixed_ds), replacement=True)
    train_loader = DataLoader(mixed_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=(args.num_workers > 0))

    print("[3/5] Loading eval sets…")
    pov_eval = load_pov_test(stride=1)
    aria_eval = load_aria_val()
    print(f"  POV n={len(pov_eval)}  Aria n={len(aria_eval)}")

    optim = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad],
                              lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()

    print("[4/5] Initial eval…")
    pov_agg = evaluate(model, pov_eval, "pov_test")
    aria_agg = evaluate(model, aria_eval, "aria_val")
    print(f"  baseline POV: {pov_agg.get('pov_native_pa_mpjpe_mm', float('nan')):.2f}  Aria: {aria_agg.get('aria_native_pa_mpjpe_mm', float('nan')):.2f}")
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
            kp3d_op = batch["kp3d_op"].cuda(non_blocking=True)
            with autocast(dtype=torch.bfloat16):
                out = model({"img": imgs}, {}, {}, "test")
                pred_3d = out["joints_coord_cam"]  # (B, 21, 3) OP order
                loss = keypoint_3d_l1(pred_3d, kp3d_op, root_idx=0)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(
                [p_ for p_ in model.parameters() if p_.requires_grad], 1.0)
            scaler.step(optim)
            scaler.update()
            pbar.set_postfix(loss=f"{float(loss):.4f}")
            step += 1

            if step % args.eval_every_steps == 0:
                print(f"\n[step {step}] eval…")
                pov_agg = evaluate(model, pov_eval, "pov_test")
                aria_agg = evaluate(model, aria_eval, "aria_val")
                pov_pa = pov_agg.get("pov_native_pa_mpjpe_mm", float("nan"))
                aria_pa = aria_agg.get("aria_native_pa_mpjpe_mm", float("nan"))
                hsam_pa = aria_agg.get("hsam_pa_mpjpe_mm", float("nan"))
                print(f"  POV: {pov_pa:.2f}  Aria: {aria_pa:.2f}  HSAM: {hsam_pa:.2f}")
                if aria_pa < best_aria:
                    best_aria = aria_pa
                    torch.save({"step": step, "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "pov_pa_mpjpe": pov_pa, "aria_pa_mpjpe": aria_pa},
                               f"{args.out_dir}/honet_ft_best_aria.pth")
                    print(f"  ⭐ new best Aria: {aria_pa:.2f}")
                with open(f"{args.out_dir}/eval_log.jsonl", "a") as f:
                    f.write(json.dumps({
                        "step": step, "epoch": epoch,
                        "pov_pa_mpjpe": pov_pa, "aria_pa_mpjpe": aria_pa,
                        "aria_hsam_pa_mpjpe": hsam_pa,
                    }) + "\n")

    print(f"\n[done]  best Aria: {best_aria:.2f}")
    pov_agg = evaluate(model, pov_eval, "pov_test")
    aria_agg = evaluate(model, aria_eval, "aria_val")
    print(f"FINAL, POV: {pov_agg.get('pov_native_pa_mpjpe_mm', float('nan')):.2f}  Aria: {aria_agg.get('aria_native_pa_mpjpe_mm', float('nan')):.2f}")
    torch.save({"step": step, "model_state_dict": model.state_dict()},
               f"{args.out_dir}/honet_ft_final.pth")


if __name__ == "__main__":
    main()
