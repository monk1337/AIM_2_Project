"""Per-sequence mesh-overlay viz: cherry-pick the best Aria sequence(s) and
render off-shelf vs FT'd mesh overlays for a strided subset of frames.

Builds on viz_mesh_overlay.py: same alignment + rendering, but filtered to
samples from a chosen sequence. Useful for storytelling, pick the sequence
where FT shows the largest improvement and show ~5-10 frames as a strip.
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from eval_joint_orders import MANO_TO_OPENPOSE
from eval_metrics import metrics_3d, procrustes_align
from torch.amp import autocast

OP_VALID = [0] + list(range(2, 21))
COLOR = (240, 200, 250)
ALPHA = 0.65


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


def fit_2d_similarity(pred_kp3d, gt_kp_2d, force_det=None):
    """2D similarity (rot+scale+trans) from pred_kp3d xy → gt_kp_2d.
    If force_det is +1 or -1, constrain R to that determinant; else pick natural sign."""
    p = pred_kp3d[:, :2].astype(np.float64)
    g = gt_kp_2d.astype(np.float64)
    p_mean, g_mean = p.mean(0), g.mean(0)
    pc, gc = p - p_mean, g - g_mean
    H = pc.T @ gc
    U, S, Vt = np.linalg.svd(H)
    if force_det is None:
        d = np.sign(np.linalg.det(Vt.T @ U.T))
    else:
        d = float(force_det)
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    s = (S * np.diag(D)).sum() / ((pc * pc).sum() + 1e-12)
    t = g_mean - s * R @ p_mean
    return float(s), R.astype(np.float32), t.astype(np.float32)


def pick_better_orientation(pred_verts_aligned, pred_kp_aligned, gt_verts, gt_kp_2d):
    """Fix B: choose the 2D-similarity orientation (det=+1 or det=-1) by
    scale-normalized mesh agreement. After 3D PA, predicted ≈ GT in 3D; we
    project both predicted and GT meshes through each candidate similarity and
    compare their 2D layout. To avoid the small-scale bias (where det=-1 can
    win by collapsing the projection), the residual is normalized by the
    candidate scale s.
    """
    best = None
    for d2 in (+1, -1):
        s, R, t = fit_2d_similarity(pred_kp_aligned, gt_kp_2d, force_det=d2)
        v_pred_2d = (s * (R @ pred_verts_aligned[:, :2].T)).T + t
        v_gt_2d   = (s * (R @ gt_verts[:, :2].T)).T + t
        score = float(np.linalg.norm(v_pred_2d - v_gt_2d, axis=1).mean()) / max(abs(s), 1e-9)
        if best is None or score < best[0]:
            best = (score, s, R, t)
    _, s, R, t = best
    return s, R, t


def project_verts(verts, s, R, t):
    out = np.empty_like(verts, dtype=np.float32)
    out[:, :2] = (s * (R @ verts[:, :2].T)).T + t
    out[:, 2] = verts[:, 2] * s
    return out


def pa_align_mesh(pred_kp_3d, pred_verts, gt_kp_3d):
    _, R, s, t = procrustes_align(pred_kp_3d, gt_kp_3d)
    aligned_kp = (s * (R @ pred_kp_3d.T)).T + t
    aligned_v = (s * (R @ pred_verts.T)).T + t
    return aligned_kp, aligned_v


def err_to_bgr(err_mm, vmax=15.0):
    """Map per-vertex error (mm) to BGR. 0=green, vmax/2=yellow, vmax+=red."""
    t = np.clip(err_mm / vmax, 0.0, 1.0)
    # Two-stop colormap: green (0,255,0) → yellow (0,255,255) → red (0,0,255) in BGR
    r = np.where(t < 0.5, 0.0, (t - 0.5) * 2)
    g = np.where(t < 0.5, 1.0, 1.0 - (t - 0.5) * 2)
    b = np.zeros_like(t)
    # Blend toward warm → BGR
    bgr = np.stack([b * 255, g * 255, r * 255 + (1 - r) * 60], axis=-1)
    return np.clip(bgr, 0, 255)


def render_mesh_overlay(crop_bgr, verts_proj, faces, color=COLOR, alpha=ALPHA,
                        per_vertex_err_mm=None, err_vmax=15.0):
    H, W = crop_bgr.shape[:2]
    if not np.isfinite(verts_proj).all():
        return crop_bgr
    verts_proj = verts_proj.copy()
    verts_proj[:, 0] = np.clip(verts_proj[:, 0], -W, 2 * W)
    verts_proj[:, 1] = np.clip(verts_proj[:, 1], -H, 2 * H)
    mask = np.zeros((H, W), dtype=np.uint8)
    face_z = verts_proj[faces, 2].mean(axis=1)
    order = np.argsort(-face_z)
    v0 = verts_proj[faces[:, 0]]; v1 = verts_proj[faces[:, 1]]; v2 = verts_proj[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    nn = np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    n = n / nn
    light = np.array([0.4, 0.4, -1.0], dtype=np.float32)
    light = light / np.linalg.norm(light)
    intensity = np.clip(0.45 + 0.55 * np.abs(n @ light), 0.0, 1.0).astype(np.float32)

    # Per-face color: error heatmap if available, else solid color
    if per_vertex_err_mm is not None:
        v_bgr = err_to_bgr(per_vertex_err_mm, vmax=err_vmax)  # (V, 3) in BGR 0..255
        face_color = v_bgr[faces].mean(axis=1)  # (F, 3)
    else:
        face_color = np.tile(np.array(color, dtype=np.float32), (faces.shape[0], 1))

    canvas = crop_bgr.astype(np.float32).copy()
    for i in order:
        pts = verts_proj[faces[i], :2]
        if pts[:, 0].max() < 0 or pts[:, 1].max() < 0 or pts[:, 0].min() > W or pts[:, 1].min() > H:
            continue
        pts_i = np.round(pts).astype(np.int32)
        sh = intensity[i]
        c = (face_color[i] * sh).tolist()
        cv2.fillConvexPoly(canvas, pts_i, c)
        cv2.fillConvexPoly(mask, pts_i, (255,))
    m = (mask > 0).astype(np.float32)[..., None]
    out = canvas * (alpha * m) + crop_bgr.astype(np.float32) * (1.0 - alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def label_panel(img_rgb, label, sub=None):
    pil = Image.fromarray(img_rgb)
    d = ImageDraw.Draw(pil)
    try:
        f1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        f2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        f1 = f2 = None
    H = 22 if sub is None else 38
    d.rectangle([0, 0, 256, H], fill=(0, 0, 0))
    d.text((4, 2), label, fill=(255, 255, 255), font=f1)
    if sub:
        d.text((4, 20), sub, fill=(200, 200, 200), font=f2)
    return np.array(pil)


def run_batch(model, crops, device="cuda"):
    crops_bgr = np.stack([c[:, :, ::-1].astype(np.float32) for c in crops])
    x = torch.from_numpy(crops_bgr).to(device, dtype=torch.float32)
    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
        out = model(x)
    return out


def compute_pa(p3d, samp, kind, gt_choice="hsam"):
    """Aria: gt_choice in {'hsam','mps'}. POV: native synthetic."""
    pred_op = p3d[MANO_TO_OPENPOSE]
    if kind == "aria":
        if gt_choice == "mps":
            gt_op = samp["aria_eval_joints_3d_op"]
            return metrics_3d(pred_op, gt_op, joint_subset=OP_VALID)["pa_mpjpe_mm"] / 1000.0
        # default: HSAM (HaMeR pseudo-label, MANO order in samp)
        gt_op = samp["native_joints_3d"][MANO_TO_OPENPOSE]
        return metrics_3d(pred_op, gt_op)["pa_mpjpe_mm"] / 1000.0
    gt_op = samp["native_joints_3d"][MANO_TO_OPENPOSE]
    return metrics_3d(pred_op, gt_op)["pa_mpjpe_mm"] / 1000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ft_ckpt", required=True)
    p.add_argument("--seqs_aria", nargs="*", default=[])
    p.add_argument("--seqs_pov", nargs="*", default=[])
    p.add_argument("--out_dir", default="/workspace/results/viz_seq_mesh")
    p.add_argument("--max_per_seq", type=int, default=12,
                   help="render at most this many frames per sequence (strided uniformly)")
    p.add_argument("--aria_gt", default="hsam", choices=["hsam", "mps"],
                   help="which Aria GT to compute PA-MPJPE against in the subtitle")
    p.add_argument("--wins_from", default=None,
                   help="Path to per_seq JSON. If set, picks the TOP --max_per_seq frames "
                        "with biggest improvement (off - ft) per sequence instead of stride-sampling.")
    p.add_argument("--min_off_mm", type=float, default=0.0,
                   help="When using --wins_from, only include frames whose OFF-SHELF error "
                        "is >= this many mm (so we showcase off-shelf failures specifically).")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("[1/3] Loading models...")
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    pipe_off = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
    model_off = pipe_off.wilor_model.eval()
    pipe_ft = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
    ck = torch.load(args.ft_ckpt, map_location="cuda", weights_only=False)
    pipe_ft.wilor_model.load_state_dict(ck["model_state_dict"])
    model_ft = pipe_ft.wilor_model.eval()
    faces = np.array(model_off.mano.faces, dtype=np.int64)
    print(f"  faces={faces.shape}")

    print("[2/3] Loading samples + filtering by sequence...")
    aria = load_aria_val()
    pov = load_pov_test(stride=1)

    wins = None
    if args.wins_from:
        wj = json.load(open(args.wins_from))
        # Build per-sample dicts keyed by (kind, idx) → off/ft mm.
        # JSON has "aria_off"/"aria_ft" lists with per-row {idx, seq, hsam_mm or mps_mm}.
        def _key(metric_keys, rows):
            out = {}
            for r in rows:
                for mk in metric_keys:
                    if mk in r:
                        out[r["idx"]] = (r["seq"], r[mk]); break
            return out
        a_off = _key(("hsam_mm", "mps_mm"), wj["aria_off"])
        a_ft = _key(("hsam_mm", "mps_mm"), wj["aria_ft"])
        p_off = _key(("pov_mm",), wj["pov_off"])
        p_ft = _key(("pov_mm",), wj["pov_ft"])
        wins = {"aria": [], "pov": []}
        for idx, (seq, om) in a_off.items():
            if idx in a_ft:
                wins["aria"].append((idx, seq, om, a_ft[idx][1]))
        for idx, (seq, om) in p_off.items():
            if idx in p_ft:
                wins["pov"].append((idx, seq, om, p_ft[idx][1]))

    samples_to_render = []
    for kind, ds, seqs in [("aria", aria, args.seqs_aria), ("pov", pov, args.seqs_pov)]:
        for seq in seqs:
            if wins is not None:
                cand = [w for w in wins[kind] if w[1] == seq and w[2] >= args.min_off_mm]
                cand.sort(key=lambda w: -(w[2] - w[3]))  # biggest off-ft delta first
                cand = cand[:args.max_per_seq]
                ix = [c[0] for c in cand]
                print(f"  {kind} {seq}: top {len(ix)} wins by off-ft delta "
                      f"(min off >= {args.min_off_mm:.0f} mm)")
            else:
                ix = [i for i, s in enumerate(ds) if s["sequence_name"] == seq]
                if not ix:
                    print(f"  WARN no samples found for {kind}:{seq}")
                    continue
                if len(ix) > args.max_per_seq:
                    step = len(ix) // args.max_per_seq
                    ix = ix[::step][:args.max_per_seq]
                print(f"  {kind} {seq}: {len(ix)} stride-sampled frames")
            for i in ix:
                samples_to_render.append((kind, seq, ds[i]))

    print("[3/3] Rendering...")
    metrics = []
    for n, (kind, seq, samp) in enumerate(tqdm(samples_to_render)):
        img = np.asarray(get_image(samp))
        bbox = derive_bbox_from_joints2d(samp["native_joints_2d"], padding=1.5,
                                         img_wh=samp["image_wh"])
        crop, flip, M = crop_for_wilor(img, bbox, samp["hand_side"])
        gt_kp = gt_2d_in_crop(samp, M, flip, img.shape[1])

        out_off = run_batch(model_off, [crop])
        out_ft = run_batch(model_ft, [crop])
        v_off = out_off["pred_vertices"][0].float().cpu().numpy()
        v_ft = out_ft["pred_vertices"][0].float().cpu().numpy()
        kp_off = out_off["pred_keypoints_3d"][0].float().cpu().numpy()
        kp_ft = out_ft["pred_keypoints_3d"][0].float().cpu().numpy()
        # crop_for_wilor flips left hands → right canonical for the model.
        # Flip pred xs back so they live in the original (GT) frame.
        if flip:
            v_off = v_off.copy(); v_off[:, 0] = -v_off[:, 0]
            v_ft = v_ft.copy(); v_ft[:, 0] = -v_ft[:, 0]
            kp_off = kp_off.copy(); kp_off[:, 0] = -kp_off[:, 0]
            kp_ft = kp_ft.copy(); kp_ft[:, 0] = -kp_ft[:, 0]
        pa_off = compute_pa(kp_off, samp, kind, gt_choice=args.aria_gt)
        pa_ft = compute_pa(kp_ft, samp, kind, gt_choice=args.aria_gt)

        gt3d = samp["native_joints_3d"]
        kp_off_a, v_off_a = pa_align_mesh(kp_off, v_off, gt3d)
        kp_ft_a, v_ft_a = pa_align_mesh(kp_ft, v_ft, gt3d)
        # Fix B: pick image-plane orientation by scale-normalized mesh agreement
        # against GT mesh. Falls back to the natural 2D fit if GT mesh is missing.
        gt_v_for_fix = samp.get("native_vertices")
        if gt_v_for_fix is not None and gt_v_for_fix.shape == v_off.shape:
            s_off, R_off, t_off = pick_better_orientation(v_off_a, kp_off_a, gt_v_for_fix, gt_kp)
            s_ft,  R_ft,  t_ft  = pick_better_orientation(v_ft_a,  kp_ft_a,  gt_v_for_fix, gt_kp)
        else:
            s_off, R_off, t_off = fit_2d_similarity(kp_off_a, gt_kp)
            s_ft,  R_ft,  t_ft  = fit_2d_similarity(kp_ft_a,  gt_kp)
        verts_off = project_verts(v_off_a, s_off, R_off, t_off)
        verts_ft = project_verts(v_ft_a, s_ft, R_ft, t_ft)

        # Per-vertex error vs GT mesh (PA-aligned to GT 3D, in mm).
        # native_vertices is HSAM/synthetic GT in (778, 3) MANO topology.
        gt_v = samp.get("native_vertices")
        per_v_err_off = per_v_err_ft = None
        if gt_v is not None and gt_v.shape == v_off.shape:
            per_v_err_off = np.linalg.norm(v_off_a - gt_v, axis=1) * 1000.0
            per_v_err_ft = np.linalg.norm(v_ft_a - gt_v, axis=1) * 1000.0

        crop_bgr = crop[:, :, ::-1].copy()
        mesh_off = render_mesh_overlay(crop_bgr, verts_off, faces,
                                       per_vertex_err_mm=per_v_err_off)[:, :, ::-1]
        mesh_ft = render_mesh_overlay(crop_bgr, verts_ft, faces,
                                      per_vertex_err_mm=per_v_err_ft)[:, :, ::-1]
        sub_off = f"PA-MPJPE: {pa_off*1000:.1f} mm"
        sub_ft = f"PA-MPJPE: {pa_ft*1000:.1f} mm"
        if per_v_err_off is not None:
            sub_off += f"  •  PVE: {per_v_err_off.mean():.1f} mm"
            sub_ft += f"  •  PVE: {per_v_err_ft.mean():.1f} mm"
        p1 = label_panel(crop, "input crop")
        p2 = label_panel(mesh_off, "off-shelf WiLoR", sub_off)
        p3 = label_panel(mesh_ft, "FT WiLoR", sub_ft)
        combined = np.zeros((256, 256 * 3 + 8, 3), dtype=np.uint8)
        combined[:, :256] = p1
        combined[:, 256 + 4:256 * 2 + 4] = p2
        combined[:, 256 * 2 + 8:] = p3
        Image.fromarray(combined).save(
            f"{args.out_dir}/{kind}_{seq}_{samp['frame_id']:06d}_{samp['hand_side']}.jpg",
            quality=88)
        metrics.append({"kind": kind, "seq": seq, "frame_id": int(samp["frame_id"]),
                        "side": samp["hand_side"],
                        "pa_off_mm": float(pa_off * 1000), "pa_ft_mm": float(pa_ft * 1000)})

    with open(f"{args.out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {len(metrics)} panels to {args.out_dir}/")


if __name__ == "__main__":
    main()
