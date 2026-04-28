"""Side-by-side viz: GT | off-shelf WiLoR | distilled WiLoR.

100 samples (50 Aria val + 50 POV test), saved as
/workspace/results/viz_compare/{aria,pov}_NNN.jpg with 3-panel layout.
"""
import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from eval_joint_orders import MANO_TO_OPENPOSE


OUT_DIR = "/workspace/results/viz_compare"
FT_CKPT = "/workspace/checkpoints/wilor_ft_distill/wilor_ft_final.pth"

# MANO bone connectivity (parent indices)
MANO_PARENTS = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]


def crop_for_wilor(img, bbox, hand_side, image_size=256):
    is_right = 1 if hand_side == "right" else 0
    flip = (is_right == 0)
    if flip:
        img = img[:, ::-1].copy()
        bbox = bbox.copy()
        bbox[0] = img.shape[1] - bbox[0] - bbox[2]
    x_, y_, w_, h_ = bbox
    cx, cy = x_ + w_ / 2, y_ + h_ / 2
    bsize = max(w_, h_)
    src = np.array([[cx - bsize / 2, cy - bsize / 2],
                    [cx + bsize / 2, cy - bsize / 2],
                    [cx - bsize / 2, cy + bsize / 2]], dtype=np.float32)
    dst = np.array([[0, 0], [image_size, 0], [0, image_size]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(img, M, (image_size, image_size), flags=cv2.INTER_LINEAR)
    return crop, flip, M, bsize


def project_pred_2d(pred_kp_2d, image_size=256):
    """WiLoR pred_keypoints_2d is in normalized crop frame [-0.5, 0.5]."""
    return (pred_kp_2d + 0.5) * image_size


def draw_hand(img_pil, kp_2d, color, line_w=1, dot_r=2):
    d = ImageDraw.Draw(img_pil)
    for k, p in enumerate(MANO_PARENTS):
        if k == p:
            continue
        x1, y1 = kp_2d[k]
        x2, y2 = kp_2d[p]
        d.line([(x1, y1), (x2, y2)], fill=color, width=line_w)
    for x, y in kp_2d:
        d.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r],
                  fill=color, outline=(0, 0, 0))


def make_panel(crop_pil, kp_2d, label, color):
    out = crop_pil.copy()
    draw_hand(out, kp_2d, color)
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = None
    d.rectangle([0, 0, 256, 20], fill=(0, 0, 0))
    d.text((4, 2), label, fill=color, font=font)
    return out


def gt_2d_in_crop(samp, M, flip, img_w):
    j = samp["native_joints_2d"].copy()
    if flip:
        j[:, 0] = img_w - j[:, 0]
    jh = np.concatenate([j, np.ones((21, 1))], axis=1)
    return (M @ jh.T).T  # (21, 2)


def pred_2d_anchored(out_dict, j_idx, gt_kp_in_crop, root_idx=0):
    """Anchor pred wrist to GT wrist, scale pred so its xy extent = GT 2D extent.

    This gives a fair *shape* comparison overlaid on the actual hand, invariant
    to the model's absolute camera prediction (which WiLoR doesn't return here).
    """
    p3d = out_dict["pred_keypoints_3d"][j_idx].float().cpu().numpy()
    p3d_root = p3d - p3d[root_idx:root_idx + 1]
    pred_extent = np.linalg.norm(p3d_root[:, :2], axis=-1).max() + 1e-6
    gt_root = gt_kp_in_crop - gt_kp_in_crop[root_idx:root_idx + 1]
    gt_extent = np.linalg.norm(gt_root, axis=-1).max() + 1e-6
    scale = gt_extent / pred_extent
    kp = p3d_root[:, :2] * scale + gt_kp_in_crop[root_idx:root_idx + 1]
    return kp


def run_batch(model, crops, device="cuda"):
    from torch.cuda.amp import autocast
    crops_bgr = np.stack([c[:, :, ::-1].astype(np.float32) for c in crops])  # RGB→BGR
    x = torch.from_numpy(crops_bgr).to(device, dtype=torch.float32)
    with torch.no_grad(), autocast(dtype=torch.bfloat16):
        out = model(x)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/4] Loading WiLoR off-shelf + distilled...")
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    pipe_off = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
    model_off = pipe_off.wilor_model
    model_off.eval()

    pipe_ft = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
    ck = torch.load(FT_CKPT, map_location="cuda", weights_only=False)
    pipe_ft.wilor_model.load_state_dict(ck["model_state_dict"])
    model_ft = pipe_ft.wilor_model
    model_ft.eval()

    # Quick probe, what keys does the model return?
    with torch.no_grad():
        probe = model_off(torch.zeros(1, 256, 256, 3, device="cuda"))
    print(f"  WiLoR output keys: {list(probe.keys())}")

    print("[2/4] Loading samples (50 Aria + 50 POV)...")
    aria = load_aria_val()
    pov = load_pov_test(stride=1)
    rng = np.random.default_rng(42)
    aria_idx = rng.choice(len(aria), 50, replace=False)
    pov_idx = rng.choice(len(pov), 50, replace=False)
    aria_samples = [aria[i] for i in aria_idx]
    pov_samples = [pov[i] for i in pov_idx]

    for kind, samples in [("aria", aria_samples), ("pov", pov_samples)]:
        print(f"[3/4] {kind} ({len(samples)} samples)...")
        for n, samp in enumerate(tqdm(samples, desc=kind)):
            img = np.asarray(get_image(samp))
            bbox = derive_bbox_from_joints2d(samp["native_joints_2d"], padding=1.5,
                                             img_wh=samp["image_wh"])
            crop, flip, M, bsize = crop_for_wilor(img, bbox, samp["hand_side"])
            gt_kp = gt_2d_in_crop(samp, M, flip, img.shape[1])
            # Both models in one batch (both eval, no grad)
            out_off = run_batch(model_off, [crop])
            out_ft = run_batch(model_ft, [crop])
            kp_off = pred_2d_anchored(out_off, 0, gt_kp)
            kp_ft = pred_2d_anchored(out_ft, 0, gt_kp)
            # GT in canonical right frame too, `crop_for_wilor` flipped image already
            # so GT in `crop` coords IS in canonical right (no extra flip needed)
            crop_pil = Image.fromarray(crop)
            p1 = make_panel(crop_pil, gt_kp, "GT", (0, 255, 0))
            p2 = make_panel(crop_pil, kp_off, "off-shelf WiLoR", (255, 80, 80))
            p3 = make_panel(crop_pil, kp_ft, "distilled WiLoR", (80, 160, 255))
            combined = Image.new("RGB", (256 * 3 + 8, 256), (0, 0, 0))
            combined.paste(p1, (0, 0))
            combined.paste(p2, (256 + 4, 0))
            combined.paste(p3, (256 * 2 + 8, 0))
            seq = samp["sequence_name"]
            fid = samp["frame_id"]
            side = samp["hand_side"]
            combined.save(f"{OUT_DIR}/{kind}_{n:03d}_{seq}_{fid}_{side}.jpg", quality=85)

    print(f"[4/4] Done. Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
