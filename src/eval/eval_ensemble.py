"""Ensemble eval: average WiLoR-FT (best Aria) + HandOccNet off-shelf predictions."""
import os
import sys
import json
import numpy as np
import torch
import cv2
from tqdm import tqdm

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from eval_joint_orders import mano_to_op
from eval_metrics import metrics_3d, metrics_pve, aggregate, procrustes_align


# Wilor best Aria checkpoint
WILOR_CKPT = "/workspace/checkpoints/wilor_ft_aria4x/wilor_ft_best_aria.pth"


def crop_with_wilor(pipe, image_np, bbox_xywh, is_right):
    from wilor_mini.utils import utils
    from skimage.filters import gaussian
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2.0, y + h / 2.0
    bbox_size = max(w, h)
    flip = (is_right == 0)
    patch_size = pipe.IMAGE_SIZE
    cvimg = image_np.copy()
    ds = (bbox_size / patch_size) / 2.0
    if ds > 1.1:
        cvimg = gaussian(cvimg, sigma=(ds - 1) / 2, channel_axis=2, preserve_range=True)
    img_patch_cv, _ = utils.generate_image_patch_cv2(
        cvimg, cx, cy, bbox_size, bbox_size, patch_size, patch_size,
        flip, 1.0, 0, border_mode=cv2.BORDER_CONSTANT)
    return img_patch_cv, flip


def predict_wilor(rows, pipe):
    from wilor_mini.utils import utils as wutils
    out = {}
    for r in tqdm(rows, desc="WiLoR-FT"):
        img = np.asarray(get_image(r))
        bbox = derive_bbox_from_joints2d(r["native_joints_2d"], padding=1.5, img_wh=r["image_wh"])
        is_right = 1 if r["hand_side"] == "right" else 0
        patch_bgr, flip = crop_with_wilor(pipe, img, bbox, is_right)
        # patch_bgr is uint8 BGR (H, W, 3); wilor model takes (B, H, W, 3)
        x = torch.from_numpy(patch_bgr[None]).to("cuda", dtype=torch.float32)
        with torch.no_grad():
            o = pipe.wilor_model(x)
        pred_3d = o["pred_keypoints_3d"][0].cpu().float().numpy().copy()
        if flip:
            pred_3d[:, 0] = -pred_3d[:, 0]
        out[(r["sequence_name"], r["frame_id"], r["hand_side"])] = mano_to_op(pred_3d) - mano_to_op(pred_3d)[0:1]
    return out


def predict_handoccnet(rows, model, batch_size=64):
    from torch.cuda.amp import autocast
    out = {}
    patches, meta = [], []
    for r in rows:
        img = np.asarray(get_image(r))
        bbox = derive_bbox_from_joints2d(r["native_joints_2d"], padding=1.5, img_wh=r["image_wh"])
        is_right = 1 if r["hand_side"] == "right" else 0
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
        patches.append(patch)
        meta.append({"row": r, "is_right": is_right})

    bs = batch_size
    for i in tqdm(range(0, len(patches), bs), desc="HONet"):
        x = torch.from_numpy(np.stack(patches[i:i+bs])).cuda()
        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            o = model({"img": x}, {}, {}, "test")
        joints = o["joints_coord_cam"].float().cpu().numpy()
        for j, m in enumerate(meta[i:i+bs]):
            r = m["row"]
            p3d = joints[j].copy()
            if not m["is_right"]:
                p3d[:, 0] = -p3d[:, 0]
            p3d = p3d - p3d[0:1]
            out[(r["sequence_name"], r["frame_id"], r["hand_side"])] = p3d
    return out


def main():
    print("[1/4] Loading datasets…")
    aria_eval = load_aria_val()
    pov_eval = load_pov_test(stride=1)
    print(f"  Aria n={len(aria_eval)}  POV n={len(pov_eval)}")

    print("[2/4] Loading models…")
    # WiLoR with best-Aria FT weights
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    pipe = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
    if os.path.exists(WILOR_CKPT):
        ck = torch.load(WILOR_CKPT, map_location="cuda", weights_only=False)
        pipe.wilor_model.load_state_dict(ck["model_state_dict"], strict=False)
        print(f"  loaded WiLoR FT from {WILOR_CKPT}")
    pipe.wilor_model.eval()

    # HandOccNet
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

    print("[3/4] Predicting on Aria val (n=2333)…")
    wilor_aria = predict_wilor(aria_eval, pipe)
    honet_aria = predict_handoccnet(aria_eval, honet_model)

    print("[4/4] Computing ensemble metrics on Aria…")
    from eval_runner import compute_metrics, run_metrics_aggregation
    samples = []
    for r in aria_eval:
        key = (r["sequence_name"], r["frame_id"], r["hand_side"])
        wp = wilor_aria.get(key)
        hp = honet_aria.get(key)
        if wp is None or hp is None:
            continue
        # Average in OP order (both already in OP)
        avg = (wp + hp) / 2.0
        samples.append({
            "row": r,
            "pred_3d_mano": avg,  # actually OP order
            "pred_2d_mano": np.zeros((21, 2), dtype=np.float32),
            "pred_verts_mano": np.zeros((778, 3), dtype=np.float32),
            "pred_cam_t_full": np.zeros(3, dtype=np.float32),
        })
    samples = compute_metrics(samples, "aria_val", root_aligned_model=True, pred_in_op_order=True)
    agg, _ = run_metrics_aggregation(samples, "aria_val", "Ensemble (WiLoR-FT + HONet)", True)
    print("\n=== Ensemble on Aria val ===")
    for k in ["aria_native_pa_mpjpe_mm", "hsam_pa_mpjpe_mm", "n"]:
        v = agg.get(k)
        if v is not None and isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    with open("/workspace/results/ensemble_aria.json", "w") as f:
        json.dump(agg, f, default=float)


if __name__ == "__main__":
    main()
