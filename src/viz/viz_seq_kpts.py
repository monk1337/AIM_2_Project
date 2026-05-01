"""Render dual-skeleton overlays (off-shelf vs FT) for the 6 curated frames.
Both skeletons drawn on the SAME crop in two distinguishable uniform colors:
  off-shelf = orange/yellow,  FT = sky-blue.
Output: /workspace/results/viz_kpts/<seq>_<frame>_<side>.jpg  (520x256 RGB).
"""
import os
import sys
import argparse
import numpy as np
import torch
import cv2

sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from eval_joint_orders import MANO_TO_OPENPOSE
from viz_seq_mesh import (
    crop_for_wilor, gt_2d_in_crop, fit_2d_similarity, label_panel, compute_pa,
)
from torch.amp import autocast

BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

OFF_COLOR = (60, 180, 255)    # BGR, orange/yellow
FT_COLOR  = (240, 170, 80)    # BGR, sky-blue
GT_COLOR  = (80, 220, 80)     # BGR, green


def draw_one_skeleton(canvas, kp_2d, color, line_thickness=3, dot_radius=5):
    for a, b in BONES:
        pa = tuple(kp_2d[a].astype(int))
        pb = tuple(kp_2d[b].astype(int))
        cv2.line(canvas, pa, pb, color, line_thickness, lineType=cv2.LINE_AA)
    for j in kp_2d:
        cv2.circle(canvas, tuple(j.astype(int)), dot_radius, color, -1, lineType=cv2.LINE_AA)
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ft_ckpt", required=True)
    p.add_argument("--out_dir", default="/workspace/results/viz_kpts")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    SEL = [
        ("aria", "PR81", 252, "left"),
        ("aria", "PR82", 256, "left"),
        ("aria", "PR84", 159, "left"),
        ("pov",  "R2_r_diskplacer_1", 484, "right"),
        ("pov",  "R2_r_diskplacer_1", 681, "right"),
        ("pov",  "R2_s_scalpel_1",   1600, "right"),
    ]

    print("[1/3] Loading models…")
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    pipe_off = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
    model_off = pipe_off.wilor_model.eval()
    pipe_ft = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
    ck = torch.load(args.ft_ckpt, map_location="cuda", weights_only=False)
    pipe_ft.wilor_model.load_state_dict(ck["model_state_dict"])
    model_ft = pipe_ft.wilor_model.eval()

    print("[2/3] Loading samples…")
    aria = {(s["sequence_name"], int(s["frame_id"]), s["hand_side"]): s
            for s in load_aria_val()}
    pov = {(s["sequence_name"], int(s["frame_id"]), s["hand_side"]): s
           for s in load_pov_test(stride=1)}

    print("[3/3] Rendering…")
    for kind, seq, fid, side in SEL:
        key = (seq, fid, side)
        samp = aria.get(key) if kind == "aria" else pov.get(key)
        if samp is None:
            print(f"  miss: {kind} {seq} {fid} {side}")
            continue
        img = np.asarray(get_image(samp))
        bbox = derive_bbox_from_joints2d(samp["native_joints_2d"], padding=1.5,
                                         img_wh=samp["image_wh"])
        crop_bgr_rgb, flip, M = crop_for_wilor(img, bbox, side)
        crop_bgr = crop_bgr_rgb[:, :, ::-1].astype(np.float32)

        # gt_2d_in_crop returns joints in native (MANO) order, convert to OP so
        # indices line up with kp_op for the Procrustes fit and OP-edges drawing.
        gt_kp_2d_crop_mano = gt_2d_in_crop(samp, M, flip, img.shape[1])
        gt_kp_2d_crop = gt_kp_2d_crop_mano[MANO_TO_OPENPOSE]

        x = torch.from_numpy(crop_bgr[None]).cuda().float()
        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            r_off = model_off(x)
            r_ft = model_ft(x)
        kp_off = r_off["pred_keypoints_3d"][0].float().cpu().numpy()
        kp_ft = r_ft["pred_keypoints_3d"][0].float().cpu().numpy()
        if flip:
            kp_off = kp_off.copy(); kp_off[:, 0] = -kp_off[:, 0]
            kp_ft = kp_ft.copy(); kp_ft[:, 0] = -kp_ft[:, 0]
        kp_off_op = kp_off[MANO_TO_OPENPOSE]
        kp_ft_op = kp_ft[MANO_TO_OPENPOSE]

        # Per-skeleton: scale by 3D palm length (real physical), rotation from
        # palm-Procrustes against GT, place at GT centroid.
        PALM = [0, 1, 5, 9, 13, 17]
        crop_size = crop_bgr_rgb.shape[0]
        target_palm_px = crop_size * 0.50

        def project(kp_op, gt_2d):
            palm_3d = np.linalg.norm(kp_op[9] - kp_op[0])
            s_self = target_palm_px / max(palm_3d, 1e-6)
            kp = (kp_op[:, :2] - kp_op[0:1, :2]) * s_self
            # Rotation from full-21-joint Procrustes, more stable than palm-only
            # when GT 2D has fisheye distortion or other noise.
            _, R, _ = fit_2d_similarity(kp_op, gt_2d, force_det=+1)
            kp = (R @ kp.T).T
            t = gt_2d.mean(axis=0) - kp.mean(axis=0)
            return kp + t

        kp_off_2d = project(kp_off_op, gt_kp_2d_crop)
        kp_ft_2d  = project(kp_ft_op,  gt_kp_2d_crop)

        try:
            pa_off = compute_pa(kp_off_op, samp, kind, gt_choice="hsam")
            pa_ft  = compute_pa(kp_ft_op,  samp, kind, gt_choice="hsam")
        except Exception:
            pa_off = pa_ft = 0.0

        crop_rgb = crop_bgr_rgb.astype(np.uint8)
        canvas_bgr = crop_rgb[:, :, ::-1].copy()
        # upscale to 512 so thick strokes/dots read clearly
        UP = 2
        canvas_bgr = cv2.resize(canvas_bgr, (canvas_bgr.shape[1] * UP, canvas_bgr.shape[0] * UP),
                                interpolation=cv2.INTER_CUBIC)
        crop_rgb_up = cv2.resize(crop_rgb, (crop_rgb.shape[1] * UP, crop_rgb.shape[0] * UP),
                                 interpolation=cv2.INTER_CUBIC)
        # Draw GT first (background, thicker), then off-shelf, then FT on top.
        draw_one_skeleton(canvas_bgr, gt_kp_2d_crop * UP, GT_COLOR, line_thickness=4, dot_radius=5)
        draw_one_skeleton(canvas_bgr, kp_off_2d * UP, OFF_COLOR, line_thickness=3, dot_radius=4)
        draw_one_skeleton(canvas_bgr, kp_ft_2d  * UP, FT_COLOR,  line_thickness=3, dot_radius=4)
        dual_rgb = canvas_bgr[:, :, ::-1]
        H = dual_rgb.shape[0]  # 512

        gap = np.full((H, 8, 3), 0, dtype=np.uint8)
        panel = np.concatenate([
            label_panel(crop_rgb_up, "input crop"),
            gap,
            label_panel(dual_rgb, "GT (green)  off-shelf (orange)  FT (blue)"),
        ], axis=1)

        out_path = f"{args.out_dir}/{kind}_{seq}_{fid:06d}_{side}.jpg"
        cv2.imwrite(out_path, panel[:, :, ::-1])
        print(f"  saved {out_path}  off={pa_off:.1f}  ft={pa_ft:.1f}")

    print("done")


if __name__ == "__main__":
    main()
