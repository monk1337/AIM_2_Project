"""3-panel mesh overlay: input | off-shelf WiLoR mesh | distilled WiLoR mesh.

Uses software rasterizer (no pyrender / OpenGL), projects MANO vertices via
weak-perspective camera fit to GT 2D keypoints, then depth-sorts triangles
with simple Lambertian shading and alpha-blends onto the crop.
"""
import os
import sys
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

OP_VALID = [0] + list(range(2, 21))  # OP[1] is dummy for Aria MPS GT
from torch.cuda.amp import autocast

OUT_DIR = "/workspace/results/viz_mesh_overlay"
FT_CKPT = "/workspace/checkpoints/wilor_ft_distill/wilor_ft_final.pth"
COLOR = (240, 200, 250)  # purple-pink (BGR for cv2)
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
    return crop, flip, M, bsize


def gt_2d_in_crop(samp, M, flip, img_w):
    j = samp["native_joints_2d"].copy()
    if flip:
        j[:, 0] = img_w - j[:, 0]
    jh = np.concatenate([j, np.ones((21, 1))], axis=1)
    return (M @ jh.T).T  # (21, 2)


def fit_2d_similarity(pred_kp3d, gt_kp_2d):
    """Closed-form 2D Procrustes fit: find s, R(2x2), t s.t. s*R@pred_xy + t ≈ gt_2d.

    Necessary when pred 3D xy axes are rotated relative to image xy (e.g. Aria HSAM
    3D has x-axis along hand, image has x-axis horizontal). Returns (s, R, t).
    """
    p = pred_kp3d[:, :2].astype(np.float64)
    g = gt_kp_2d.astype(np.float64)
    p_mean, g_mean = p.mean(0), g.mean(0)
    pc, gc = p - p_mean, g - g_mean
    H = pc.T @ gc  # (2, 2)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T  # 2x2 rotation
    s = (S * np.diag(D)).sum() / ((pc * pc).sum() + 1e-12)
    t = g_mean - s * R @ p_mean
    return float(s), R.astype(np.float32), t.astype(np.float32)


def project_verts(verts, s, R, t):
    """Apply 2D similarity to xy, isotropic scale to z. Returns (V, 3) pixel-ish."""
    out = np.empty_like(verts, dtype=np.float32)
    xy = verts[:, :2]
    out[:, :2] = (s * (R @ xy.T)).T + t
    out[:, 2] = verts[:, 2] * s
    return out


def pa_align_mesh(pred_kp_3d, pred_verts, gt_kp_3d, joint_subset=None):
    """Procrustes-align (rigid + uniform scale) pred 3D → GT 3D, applied to mesh too.

    This visualizes what PA-MPJPE measures: shape after removing rotation/scale/translation.
    The placement still leverages GT 2D in a later weak-persp fit step.
    """
    P = pred_kp_3d[joint_subset] if joint_subset is not None else pred_kp_3d
    T = gt_kp_3d[joint_subset] if joint_subset is not None else gt_kp_3d
    _, R, s, t = procrustes_align(P, T)
    aligned_kp = (s * (R @ pred_kp_3d.T)).T + t
    aligned_v = (s * (R @ pred_verts.T)).T + t
    return aligned_kp, aligned_v


def render_mesh_overlay(crop_bgr, verts_proj, faces, color=COLOR, alpha=ALPHA):
    """Software rasterize back-to-front; alpha blend onto crop_bgr."""
    H, W = crop_bgr.shape[:2]
    if not np.isfinite(verts_proj).all():
        return crop_bgr
    # Clip xy to a sane band around the image so fillConvexPoly stays fast.
    verts_proj = verts_proj.copy()
    verts_proj[:, 0] = np.clip(verts_proj[:, 0], -W, 2 * W)
    verts_proj[:, 1] = np.clip(verts_proj[:, 1], -H, 2 * H)
    mask = np.zeros((H, W), dtype=np.uint8)

    face_z = verts_proj[faces, 2].mean(axis=1)
    order = np.argsort(-face_z)  # back-to-front

    v0 = verts_proj[faces[:, 0]]
    v1 = verts_proj[faces[:, 1]]
    v2 = verts_proj[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    nn = np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    n = n / nn
    light = np.array([0.4, 0.4, -1.0], dtype=np.float32)
    light = light / np.linalg.norm(light)
    intensity = np.clip(0.45 + 0.55 * np.abs(n @ light), 0.0, 1.0).astype(np.float32)

    color_arr = np.array(color, dtype=np.float32)
    canvas = crop_bgr.astype(np.float32).copy()

    for i in order:
        pts = verts_proj[faces[i], :2]
        # Skip tris that are entirely outside crop
        if pts[:, 0].max() < 0 or pts[:, 1].max() < 0 or pts[:, 0].min() > W or pts[:, 1].min() > H:
            continue
        pts_i = np.round(pts).astype(np.int32)
        sh = intensity[i]
        c = (color_arr * sh).tolist()
        cv2.fillConvexPoly(canvas, pts_i, c)
        cv2.fillConvexPoly(mask, pts_i, (255,))

    # Alpha-blend only where the mesh was painted
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
    crops_bgr = np.stack([c[:, :, ::-1].astype(np.float32) for c in crops])  # RGB→BGR
    x = torch.from_numpy(crops_bgr).to(device, dtype=torch.float32)
    with torch.no_grad(), autocast(dtype=torch.bfloat16):
        out = model(x)
    return out


def compute_pa(p3d, samp, kind):
    """PA-MPJPE in meters.

    Aria: compare against `aria_eval_joints_3d_op` (Aria-MPS native, OP order),
          using OP_VALID 20-joint subset. This is the lab's headline metric.
    POV:  compare against `native_joints_3d` (synthetic MANO order); reorder to OP.
    """
    pred_op = p3d[MANO_TO_OPENPOSE]
    if kind == "aria":
        gt_op = samp["aria_eval_joints_3d_op"]
        return metrics_3d(pred_op, gt_op, joint_subset=OP_VALID)["pa_mpjpe_mm"] / 1000.0
    gt_op = samp["native_joints_3d"][MANO_TO_OPENPOSE]
    return metrics_3d(pred_op, gt_op)["pa_mpjpe_mm"] / 1000.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/4] Loading models...")
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    pipe_off = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
    model_off = pipe_off.wilor_model.eval()

    pipe_ft = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
    ck = torch.load(FT_CKPT, map_location="cuda", weights_only=False)
    pipe_ft.wilor_model.load_state_dict(ck["model_state_dict"])
    model_ft = pipe_ft.wilor_model.eval()

    faces = np.array(model_off.mano.faces, dtype=np.int64)  # (1538, 3)
    print(f"  MANO faces: {faces.shape}")

    print("[2/4] Loading samples...")
    aria = load_aria_val()
    pov = load_pov_test(stride=1)
    rng = np.random.default_rng(42)
    aria_idx = rng.choice(len(aria), 50, replace=False)
    pov_idx = rng.choice(len(pov), 50, replace=False)
    aria_samples = [aria[i] for i in aria_idx]
    pov_samples = [pov[i] for i in pov_idx]

    metrics = {}
    for kind, samples in [("aria", aria_samples), ("pov", pov_samples)]:
        print(f"[3/4] {kind} ({len(samples)} samples)...")
        deltas = []
        for n, samp in enumerate(tqdm(samples, desc=kind)):
            img = np.asarray(get_image(samp))
            bbox = derive_bbox_from_joints2d(samp["native_joints_2d"], padding=1.5,
                                             img_wh=samp["image_wh"])
            crop, flip, M, bsize = crop_for_wilor(img, bbox, samp["hand_side"])
            gt_kp = gt_2d_in_crop(samp, M, flip, img.shape[1])

            out_off = run_batch(model_off, [crop])
            out_ft = run_batch(model_ft, [crop])

            v_off = out_off["pred_vertices"][0].float().cpu().numpy()  # (778, 3)
            v_ft = out_ft["pred_vertices"][0].float().cpu().numpy()
            kp_off = out_off["pred_keypoints_3d"][0].float().cpu().numpy()  # (21, 3)
            kp_ft = out_ft["pred_keypoints_3d"][0].float().cpu().numpy()

            # PA-MPJPE for subtitle
            pa_off = compute_pa(kp_off, samp, kind)
            pa_ft = compute_pa(kp_ft, samp, kind)

            # Procrustes-align pred → GT 3D in *the same frame as gt_kp_2d*.
            # native_joints_3d (HSAM for aria, synthetic for pov) sits in the
            # same camera frame as native_joints_2d, which is what we use to
            # anchor the mesh in image pixels. Aligning to MPS 3D for aria
            # would put verts in a different camera frame and break the LS fit.
            gt3d_mano = samp["native_joints_3d"]  # MANO order, 21
            kp_off_a, v_off_a = pa_align_mesh(kp_off, v_off, gt3d_mano)
            kp_ft_a, v_ft_a = pa_align_mesh(kp_ft, v_ft, gt3d_mano)

            # Fit 2D similarity from PA-aligned 3D xy → GT 2D in crop frame
            # (handles rotation between camera-frame 3D axes and image axes).
            s_off, R_off, t_off = fit_2d_similarity(kp_off_a, gt_kp)
            s_ft, R_ft, t_ft = fit_2d_similarity(kp_ft_a, gt_kp)
            verts_off = project_verts(v_off_a, s_off, R_off, t_off)
            verts_ft = project_verts(v_ft_a, s_ft, R_ft, t_ft)

            # crop is RGB; convert to BGR for cv2 ops, then back
            crop_bgr = crop[:, :, ::-1].copy()
            mesh_off = render_mesh_overlay(crop_bgr, verts_off, faces)
            mesh_ft = render_mesh_overlay(crop_bgr, verts_ft, faces)
            crop_panel = crop.copy()  # RGB
            mesh_off = mesh_off[:, :, ::-1]  # BGR→RGB
            mesh_ft = mesh_ft[:, :, ::-1]

            p1 = label_panel(crop_panel, "input crop")
            p2 = label_panel(mesh_off, "off-shelf WiLoR", f"PA-MPJPE: {pa_off*1000:.1f} mm")
            p3 = label_panel(mesh_ft, "distilled WiLoR", f"PA-MPJPE: {pa_ft*1000:.1f} mm")

            combined = np.zeros((256, 256 * 3 + 8, 3), dtype=np.uint8)
            combined[:, :256] = p1
            combined[:, 256 + 4:256 * 2 + 4] = p2
            combined[:, 256 * 2 + 8:] = p3

            seq = samp["sequence_name"]
            fid = samp["frame_id"]
            side = samp["hand_side"]
            Image.fromarray(combined).save(
                f"{OUT_DIR}/{kind}_{n:03d}_{seq}_{fid}_{side}.jpg", quality=88)
            deltas.append({"n": n, "pa_off_mm": float(pa_off * 1000),
                           "pa_ft_mm": float(pa_ft * 1000),
                           "delta_mm": float((pa_off - pa_ft) * 1000)})

        d_arr = np.array([d["delta_mm"] for d in deltas])
        wins = int((d_arr > 0).sum())
        losses = int((d_arr < 0).sum())
        print(f"  {kind}: n={len(deltas)} mean off-shelf PA: "
              f"{np.mean([d['pa_off_mm'] for d in deltas]):.2f}  "
              f"distilled PA: {np.mean([d['pa_ft_mm'] for d in deltas]):.2f}  "
              f"delta: +{d_arr.mean():.2f}  W/L: {wins}/{losses}")
        metrics[kind] = {"deltas": deltas, "wins": wins, "losses": losses,
                         "mean_off_mm": float(np.mean([d['pa_off_mm'] for d in deltas])),
                         "mean_ft_mm": float(np.mean([d['pa_ft_mm'] for d in deltas]))}

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[4/4] Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
