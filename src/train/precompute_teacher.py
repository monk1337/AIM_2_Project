"""Precompute MGFM+HONet ensemble teacher on Aria train images.

For each (sequence_name, frame_id, hand_side), run both models on the GT-bbox crop
and save:
  - pred_3d_op: (21, 3) float32, ensemble-averaged (0.75 MGFM + 0.25 HONet) in OP order, root-relative
  - confidence: (21,) float32, 1 / (1 + ||MGFM - HONet||), high when teachers agree

Output: /workspace/cache/ensemble_teacher_aria_train.npz
        keyed dict {f"{seq}/{fid}/{side}": {"pred_3d_op": ..., "confidence": ...}}
"""
import os
import sys
import io
import json
import glob
import numpy as np
import torch
import cv2
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
sys.path.insert(0, "/workspace/code")
from eval_loader import derive_bbox_from_joints2d  # noqa: E402


# Override these via AIM2_ARIA_DIR, AIM2_SIDECAR_DIR, AIM2_CACHE_DIR (see SETUP.md).
import os as _os
ARIA_DIR = _os.environ.get("AIM2_ARIA_DIR", "/workspace/datasets/aria_val/data")
SIDECAR = _os.path.join(
    _os.environ.get("AIM2_SIDECAR_DIR", "/workspace/datasets/phase0_sidecars"),
    "reject_keys_all_20260419.json",
)
OUT = _os.path.join(
    _os.environ.get("AIM2_CACHE_DIR", "/workspace/cache"),
    "ensemble_teacher_aria_train.npz",
)


def ccw90_2d(j2d, W):
    return np.stack([j2d[:, 1], (W - 1) - j2d[:, 0]], axis=-1).astype(np.float32)


def crop_for_mgfm(img, bbox, hand_side):
    is_right = 1 if hand_side == "right" else 0
    if not is_right:
        img = img[:, ::-1].copy()
        bbox = bbox.copy()
        bbox[0] = img.shape[1] - bbox[0] - bbox[2]
    x_, y_, w_, h_ = bbox
    cx, cy = x_ + w_ / 2, y_ + h_ / 2
    bsize = max(w_, h_)
    src = np.array([[cx - bsize / 2, cy - bsize / 2],
                    [cx + bsize / 2, cy - bsize / 2],
                    [cx - bsize / 2, cy + bsize / 2]], dtype=np.float32)
    dst = np.array([[0, 0], [224, 0], [0, 224]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(img, M, (224, 224), flags=cv2.INTER_LINEAR)
    patch = (crop.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return patch.transpose(2, 0, 1).astype(np.float32), is_right


def crop_for_honet(img, bbox, hand_side):
    is_right = 1 if hand_side == "right" else 0
    if not is_right:
        img = img[:, ::-1].copy()
        bbox = bbox.copy()
        bbox[0] = img.shape[1] - bbox[0] - bbox[2]
    x_, y_, w_, h_ = bbox
    cx, cy = x_ + w_ / 2, y_ + h_ / 2
    bsize = max(w_, h_)
    src = np.array([[cx - bsize / 2, cy - bsize / 2],
                    [cx + bsize / 2, cy - bsize / 2],
                    [cx - bsize / 2, cy + bsize / 2]], dtype=np.float32)
    dst = np.array([[0, 0], [256, 0], [0, 256]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(img, M, (256, 256), flags=cv2.INTER_LINEAR)
    patch = (crop.astype(np.float32) / 255.0).transpose(2, 0, 1)
    return patch.astype(np.float32), is_right


def main():
    print("[1/4] Loading reject keys + parquet meta...")
    R = json.load(open(SIDECAR))
    drop = set(R.get("train_reject_keys", [])) | set(R.get("train_skip_keys", []))

    files = sorted(glob.glob(f"{ARIA_DIR}/train-*.parquet"))
    rows = []
    for f in files:
        df = pq.read_table(f).to_pandas()
        for _, r in df.iterrows():
            key = f"{r.sequence_name}/{r.frame_id}"
            if key in drop:
                continue
            rows.append({
                "seq": r.sequence_name, "fid": int(r.frame_id), "side": r.hand_side,
                "img_bytes": r.image["bytes"],
                "image_width": int(r.image_width), "image_height": int(r.image_height),
                "joints_2d_raw": np.asarray(r.joints_2d, dtype=np.float32).reshape(21, 2),
            })
    print(f"  Aria train (post-reject): {len(rows)} samples")

    print("[2/4] Loading MeshGraphormer...")
    sys.path.insert(0, "/workspace/code/MeshGraphormer")
    cwd = os.getcwd()
    os.chdir("/workspace/code/MeshGraphormer")
    from eval_meshgraphormer import build_model as mgfm_build_model
    mgfm_model, mgfm_mano, mgfm_mesh_sampler = mgfm_build_model()
    mgfm_model.eval()
    os.chdir(cwd)
    for k in list(sys.modules.keys()):
        if k.startswith("manopth") or k.startswith("manolayer"):
            del sys.modules[k]

    print("[3/4] Loading HandOccNet...")
    HONET = "/workspace/code/HandOccNet"
    sys.path.insert(0, f"{HONET}/common/utils/manopth")
    sys.path.insert(0, f"{HONET}/main")
    sys.path.insert(0, f"{HONET}/common")
    from config import cfg as honet_cfg
    honet_cfg.set_args("0")
    honet_cfg.mano_path = "/workspace/checkpoints/handoccnet/_mano_root"
    from model import get_model as honet_get_model
    from torch.nn.parallel.data_parallel import DataParallel
    honet_model = honet_get_model("test")
    honet_model = DataParallel(honet_model).cuda()
    ck = torch.load("/workspace/checkpoints/handoccnet/HandOccNet_model_dump/snapshot_demo.pth.tar",
                    map_location="cuda", weights_only=False)
    honet_model.load_state_dict(ck["network"], strict=False)
    honet_model.eval()

    print("[4/4] Predicting + ensembling on Aria train...")
    from torch.cuda.amp import autocast

    cache = {}
    BS = 32
    for i in tqdm(range(0, len(rows), BS), desc="ensemble teacher"):
        batch = rows[i:i + BS]
        imgs_np = [np.asarray(Image.open(io.BytesIO(r["img_bytes"])).convert("RGB")) for r in batch]
        # CCW90 joints_2d for bbox
        bboxes = []
        for r in batch:
            j2d = ccw90_2d(r["joints_2d_raw"], r["image_width"])
            bboxes.append(derive_bbox_from_joints2d(j2d, padding=1.5,
                                                   img_wh=(r["image_width"], r["image_height"])))

        # MGFM
        mgfm_patches, mgfm_meta = [], []
        for img, bb, r in zip(imgs_np, bboxes, batch):
            patch, is_right = crop_for_mgfm(img, bb, r["side"])
            mgfm_patches.append(patch)
            mgfm_meta.append(is_right)
        x_m = torch.from_numpy(np.stack(mgfm_patches)).to("cuda", dtype=torch.float32)
        with torch.no_grad():
            outs = mgfm_model(x_m, mgfm_mano, mgfm_mesh_sampler)
        _, p3d_m, _, _ = outs[:4]
        p3d_m = p3d_m.cpu().float().numpy()
        for j in range(len(batch)):
            p3d = p3d_m[j] - p3d_m[j, 0:1]
            if not mgfm_meta[j]:
                p3d = p3d.copy()
                p3d[:, 0] = -p3d[:, 0]
            mgfm_patches[j] = p3d  # reuse list

        # HONet
        honet_patches, honet_meta = [], []
        for img, bb, r in zip(imgs_np, bboxes, batch):
            patch, is_right = crop_for_honet(img, bb, r["side"])
            honet_patches.append(patch)
            honet_meta.append(is_right)
        x_h = torch.from_numpy(np.stack(honet_patches)).cuda()
        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            o_h = honet_model({"img": x_h}, {}, {}, "test")
        joints_h = o_h["joints_coord_cam"].float().cpu().numpy()
        for j in range(len(batch)):
            p3d = joints_h[j] - joints_h[j, 0:1]
            if not honet_meta[j]:
                p3d = p3d.copy()
                p3d[:, 0] = -p3d[:, 0]
            honet_patches[j] = p3d

        # Ensemble (0.75 MGFM + 0.25 HONet) and confidence (teacher disagreement)
        for j, r in enumerate(batch):
            mgfm_p = mgfm_patches[j]   # (21, 3) OP order, root-rel
            honet_p = honet_patches[j]  # (21, 3) OP order, root-rel
            ens = 0.75 * mgfm_p + 0.25 * honet_p
            disagree = np.linalg.norm(mgfm_p - honet_p, axis=-1)  # (21,) m
            conf = 1.0 / (1.0 + 100.0 * disagree)  # disagree=0→1.0, disagree=10mm→0.5
            key = f"{r['seq']}/{r['fid']}/{r['side']}"
            cache[key] = {
                "pred_3d_op": ens.astype(np.float32),
                "confidence": conf.astype(np.float32),
            }

    print(f"\n[5/5] Saving cache to {OUT}...")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    keys = np.array(list(cache.keys()))
    pred_3d = np.stack([cache[k]["pred_3d_op"] for k in keys])
    confidence = np.stack([cache[k]["confidence"] for k in keys])
    np.savez_compressed(OUT, keys=keys, pred_3d_op=pred_3d, confidence=confidence)
    print(f"  saved {len(cache)} entries  pred_3d shape {pred_3d.shape}  conf shape {confidence.shape}")
    print(f"  mean per-sample mean conf: {confidence.mean():.3f}  median conf: {np.median(confidence):.3f}")
    print(f"  fraction of joints w/ conf<0.5: {(confidence < 0.5).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
