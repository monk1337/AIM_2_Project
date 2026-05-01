"""4-panel viz: GT | off-shelf | distilled | distilled-PA-aligned.
Plus per-sample PA-MPJPE log + curated wins/losses figure.
"""
import os
import sys
import json
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from eval_joint_orders import MANO_TO_OPENPOSE
from eval_metrics import procrustes_align


OUT_DIR = "/workspace/results/viz_compare_v2"
FT_CKPT = "/workspace/checkpoints/wilor_ft_distill/wilor_ft_final.pth"
MANO_PARENTS = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
OP_VALID = [0] + list(range(2, 21))


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
    return crop, flip, M


def gt_2d_in_crop(samp, M, flip, img_w):
    j = samp["native_joints_2d"].copy()
    if flip:
        j[:, 0] = img_w - j[:, 0]
    jh = np.concatenate([j, np.ones((21, 1))], axis=1)
    return (M @ jh.T).T


def anchor_pred_xy(pred_xy_root, gt_kp_2d):
    pred_extent = np.linalg.norm(pred_xy_root - pred_xy_root[0:1], axis=-1).max() + 1e-6
    gt_extent = np.linalg.norm(gt_kp_2d - gt_kp_2d[0:1], axis=-1).max() + 1e-6
    scale = gt_extent / pred_extent
    return (pred_xy_root - pred_xy_root[0:1]) * scale + gt_kp_2d[0:1]


def anchor_pred_xy_safe(pred_xy_root, gt_kp_2d, ref_extent):
    """Like anchor_pred_xy, but cap scale to keep skeleton from collapsing
    when 3D xy extent is degenerate (hand pointing along depth axis)."""
    pred_extent = np.linalg.norm(pred_xy_root - pred_xy_root[0:1], axis=-1).max() + 1e-6
    gt_extent = np.linalg.norm(gt_kp_2d - gt_kp_2d[0:1], axis=-1).max() + 1e-6
    # If 3D-xy extent is collapsed relative to the unaligned pred's extent,
    # fall back to the unaligned scale (keeps viz interpretable).
    if pred_extent < 0.3 * (ref_extent + 1e-6):
        # Use GT 2D extent / the unaligned pred extent (in 3D xy units)
        scale = gt_extent / (ref_extent + 1e-6)
    else:
        scale = gt_extent / pred_extent
    return (pred_xy_root - pred_xy_root[0:1]) * scale + gt_kp_2d[0:1]


def compute_pa_mpjpe(pred_3d_mano, samp, dataset):
    """Returns (PA-MPJPE in mm, aligned_pred_mano [for viz])."""
    pred_rel = pred_3d_mano - pred_3d_mano[0:1]
    if dataset == "aria":
        gt_op = samp["aria_eval_joints_3d_op"]
        if gt_op is None:
            return None, None
        gt_op_rel = gt_op - gt_op[0:1]
        pred_op = pred_rel[MANO_TO_OPENPOSE]
        aligned_op_valid = procrustes_align(pred_op[OP_VALID], gt_op_rel[OP_VALID])[0]
        err = np.linalg.norm(aligned_op_valid - gt_op_rel[OP_VALID], axis=-1).mean() * 1000
        # Place aligned-OP back in MANO order for viz: invert MANO_TO_OPENPOSE
        op_to_mano = np.argsort(MANO_TO_OPENPOSE)
        aligned_op_full = np.zeros((21, 3), dtype=np.float32)
        aligned_op_full[OP_VALID] = aligned_op_valid
        aligned_op_full[1] = aligned_op_valid[0]   # placeholder=wrist to avoid viz outlier
        aligned_mano = aligned_op_full[op_to_mano]
        return float(err), aligned_mano
    else:  # pov
        gt = samp["native_joints_3d"]
        gt_rel = gt - gt[0:1]
        aligned = procrustes_align(pred_rel, gt_rel)[0]
        err = np.linalg.norm(aligned - gt_rel, axis=-1).mean() * 1000
        return float(err), aligned


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


def make_panel(crop_pil, kp_2d, label, color, sublabel=None):
    out = crop_pil.copy()
    draw_hand(out, kp_2d, color)
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = font_sub = None
    bar_h = 18 if not sublabel else 30
    d.rectangle([0, 0, 256, bar_h], fill=(0, 0, 0))
    d.text((4, 1), label, fill=color, font=font)
    if sublabel:
        d.text((4, 16), sublabel, fill=(220, 220, 220), font=font_sub)
    return out


def run_one(model, crop):
    from torch.cuda.amp import autocast
    crop_bgr = crop[:, :, ::-1].astype(np.float32)
    x = torch.from_numpy(crop_bgr[None]).to("cuda", dtype=torch.float32)
    with torch.no_grad(), autocast(dtype=torch.bfloat16):
        out = model(x)
    p3d = out["pred_keypoints_3d"][0].float().cpu().numpy()
    cam = out["pred_cam"][0].float().cpu().numpy() if "pred_cam" in out else None
    return p3d, cam


def project_pred_2d(pred_3d, pred_cam, image_size=256, focal=5000.0):
    """WiLoR weak-perspective: cam = [s, tx, ty]; t_z = 2f / (image_size * s)."""
    s, tx, ty = float(pred_cam[0]), float(pred_cam[1]), float(pred_cam[2])
    tz = 2.0 * focal / (image_size * s + 1e-9)
    p3d_t = pred_3d + np.array([tx, ty, tz], dtype=np.float32)
    proj = p3d_t[:, :2] / np.maximum(p3d_t[:, 2:3], 1e-6) * focal + image_size / 2.0
    return proj.astype(np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/4] Loading models...")
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    pipe_off = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
    pipe_off.wilor_model.eval()
    pipe_ft = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
    ck = torch.load(FT_CKPT, map_location="cuda", weights_only=False)
    pipe_ft.wilor_model.load_state_dict(ck["model_state_dict"])
    pipe_ft.wilor_model.eval()

    print("[2/4] Loading samples...")
    aria = load_aria_val()
    pov = load_pov_test(stride=1)
    rng = np.random.default_rng(42)
    aria_idx = rng.choice(len(aria), 50, replace=False)
    pov_idx = rng.choice(len(pov), 50, replace=False)
    aria_samples = [aria[i] for i in aria_idx]
    pov_samples = [pov[i] for i in pov_idx]

    log = []  # per-sample metrics

    for kind, samples in [("aria", aria_samples), ("pov", pov_samples)]:
        print(f"[3/4] {kind}...")
        for n, samp in enumerate(tqdm(samples, desc=kind)):
            img = np.asarray(get_image(samp))
            bbox = derive_bbox_from_joints2d(samp["native_joints_2d"], padding=1.5,
                                             img_wh=samp["image_wh"])
            crop, flip, M = crop_for_wilor(img, bbox, samp["hand_side"])
            gt_kp = gt_2d_in_crop(samp, M, flip, img.shape[1])

            p3d_off, cam_off = run_one(pipe_off.wilor_model, crop)
            p3d_ft, cam_ft = run_one(pipe_ft.wilor_model, crop)

            err_off, _ = compute_pa_mpjpe(p3d_off, samp, kind)
            err_ft, aligned_ft = compute_pa_mpjpe(p3d_ft, samp, kind)

            # Project both using off-shelf's pred_cam (distilled's cam drifts during FT since
            # it wasn't in the loss). Same camera → viz isolates the *shape* difference, which
            # is what PA-MPJPE measures.
            cam = cam_off if cam_off is not None else cam_ft
            if cam is not None:
                kp_off_xy = project_pred_2d(p3d_off, cam)
                kp_ft_xy = project_pred_2d(p3d_ft, cam)
            else:
                kp_off_xy = anchor_pred_xy(p3d_off[:, :2] - p3d_off[0:1, :2], gt_kp)
                kp_ft_xy = anchor_pred_xy(p3d_ft[:, :2] - p3d_ft[0:1, :2], gt_kp)

            crop_pil = Image.fromarray(crop)
            delta = (err_off - err_ft) if (err_off is not None and err_ft is not None) else None
            p1 = make_panel(crop_pil, gt_kp, "GT", (0, 255, 0))
            p2 = make_panel(crop_pil, kp_off_xy, "off-shelf", (255, 80, 80),
                            sublabel=f"PA-MPJPE: {err_off:.1f} mm" if err_off is not None else None)
            ft_sublabel = (f"PA-MPJPE: {err_ft:.1f} mm  (Δ {delta:+.1f})"
                           if delta is not None else
                           (f"PA-MPJPE: {err_ft:.1f} mm" if err_ft is not None else None))
            p3 = make_panel(crop_pil, kp_ft_xy, "distilled", (80, 160, 255), sublabel=ft_sublabel)
            combined = Image.new("RGB", (256 * 3 + 8, 256), (0, 0, 0))
            for i, p in enumerate([p1, p2, p3]):
                combined.paste(p, (i * (256 + 4), 0))
            seq = samp["sequence_name"]
            fid = samp["frame_id"]
            side = samp["hand_side"]
            fname = f"{kind}_{n:03d}_{seq}_{fid}_{side}.jpg"
            combined.save(f"{OUT_DIR}/{fname}", quality=85)

            log.append({
                "kind": kind, "n": n, "fname": fname,
                "seq": seq, "fid": fid, "side": side,
                "pa_mpjpe_off": err_off, "pa_mpjpe_ft": err_ft,
                "delta": (err_off - err_ft) if (err_off is not None and err_ft is not None) else None,
            })

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(log, f, indent=2)

    # Build curated wins/losses figure
    print("[4/4] Building curated figure...")
    valid = [e for e in log if e["delta"] is not None]
    for kind in ["aria", "pov"]:
        sub = [e for e in valid if e["kind"] == kind]
        sub.sort(key=lambda e: e["delta"], reverse=True)  # largest delta = biggest distilled win
        wins = sub[:6]
        losses = sub[-6:][::-1]   # largest negative delta = biggest distilled loss
        rows = []
        for tag, group in [("WINS (distilled better)", wins),
                           ("LOSSES (distilled worse)", losses)]:
            row_imgs = [Image.open(f"{OUT_DIR}/{e['fname']}") for e in group]
            row = Image.new("RGB", (row_imgs[0].width, row_imgs[0].height * len(row_imgs) + 4 * len(row_imgs)),
                            (15, 15, 15))
            for i, im in enumerate(row_imgs):
                row.paste(im, (0, i * (im.height + 4)))
            # Add header bar
            header_h = 24
            with_header = Image.new("RGB", (row.width, row.height + header_h), (60, 0, 0) if "LOSS" in tag else (0, 60, 0))
            with_header.paste(row, (0, header_h))
            d = ImageDraw.Draw(with_header)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except Exception:
                font = None
            d.text((6, 4), f"{kind.upper()}, {tag}", fill=(255, 255, 255), font=font)
            rows.append(with_header)
        # Stack wins and losses side-by-side
        H = max(r.height for r in rows)
        W = sum(r.width for r in rows) + 8
        curated = Image.new("RGB", (W, H), (0, 0, 0))
        x = 0
        for r in rows:
            curated.paste(r, (x, 0))
            x += r.width + 8
        curated.save(f"{OUT_DIR}/curated_{kind}.jpg", quality=88)
        # Print summary
        deltas = [e["delta"] for e in sub]
        offs = [e["pa_mpjpe_off"] for e in sub]
        fts = [e["pa_mpjpe_ft"] for e in sub]
        print(f"  {kind}: n={len(sub)}  off-shelf mean PA: {np.mean(offs):.2f}  "
              f"distilled mean PA: {np.mean(fts):.2f}  delta mean: {np.mean(deltas):+.2f}  "
              f"#wins({len([d for d in deltas if d > 0])}) / #losses({len([d for d in deltas if d < 0])})")

    print(f"Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
