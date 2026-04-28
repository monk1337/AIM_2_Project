"""Re-export mesh data using pa_align_mesh (joint-based alignment, then apply
to vertices), matches what viz_seq_mesh uses for the panel renders.
"""
import sys, json, numpy as np, torch
sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from viz_seq_mesh import crop_for_wilor
from eval_metrics import procrustes_align
from torch.amp import autocast

ARIA_FIDS = [104, 164, 245, 287, 377, 415]
POV_FIDS  = [179, 357, 535, 713, 891, 1959]
OUT = "/workspace/results/mesh_export_v2.json"

print("loading models")
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
pipe_off = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
m_off = pipe_off.wilor_model.eval()
pipe_ft = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
ck = torch.load("/workspace/checkpoints/wilor_ft_mixed/wilor_ft_final.pth",
                map_location="cuda", weights_only=False)
pipe_ft.wilor_model.load_state_dict(ck["model_state_dict"])
m_ft = pipe_ft.wilor_model.eval()

faces = pipe_off.wilor_model.mano.faces.astype(np.int32).tolist()

aria = {(s["sequence_name"], int(s["frame_id"]), s["hand_side"]): s
        for s in load_aria_val() if s["sequence_name"] == "PR84"}
pov = {(s["sequence_name"], int(s["frame_id"]), s["hand_side"]): s
       for s in load_pov_test(stride=1) if s["sequence_name"] == "R2_s_scalpel_1"}


def pa_align_mesh(pred_kp_3d, pred_verts, gt_kp_3d):
    """Align using keypoints, apply same transform to vertices. Matches viz_seq_mesh."""
    _, R, s, t = procrustes_align(pred_kp_3d, gt_kp_3d)
    aligned_kp = (s * (R @ pred_kp_3d.T)).T + t
    aligned_v = (s * (R @ pred_verts.T)).T + t
    return aligned_kp, aligned_v


def predict(samp, side):
    img = np.asarray(get_image(samp))
    bbox = derive_bbox_from_joints2d(samp["native_joints_2d"], padding=1.5,
                                     img_wh=samp["image_wh"])
    crop_bgr_rgb, flip, _ = crop_for_wilor(img, bbox, side)
    x = torch.from_numpy(crop_bgr_rgb[:, :, ::-1].astype(np.float32)[None]).cuda().float()
    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
        r_off = m_off(x); r_ft = m_ft(x)
    v_off = r_off["pred_vertices"][0].float().cpu().numpy()
    v_ft  = r_ft["pred_vertices"][0].float().cpu().numpy()
    kp_off = r_off["pred_keypoints_3d"][0].float().cpu().numpy()
    kp_ft  = r_ft["pred_keypoints_3d"][0].float().cpu().numpy()
    if flip:
        v_off = v_off.copy(); v_off[:, 0] = -v_off[:, 0]
        v_ft  = v_ft.copy();  v_ft[:, 0]  = -v_ft[:, 0]
        kp_off = kp_off.copy(); kp_off[:, 0] = -kp_off[:, 0]
        kp_ft  = kp_ft.copy();  kp_ft[:, 0]  = -kp_ft[:, 0]
    return kp_off, v_off, kp_ft, v_ft


def pick(seq_dict, seq, fid, prefer_side="right"):
    # Prefer the requested side; fall back to anything
    for k, s in seq_dict.items():
        if k[0] == seq and k[1] == fid and k[2] == prefer_side:
            return s, k[2]
    for k, s in seq_dict.items():
        if k[0] == seq and k[1] == fid: return s, k[2]
    return None, None


def process(seq_dict, seq_key, fids, kind):
    out = []
    for fid in fids:
        s, side = pick(seq_dict, seq_key, fid)
        if s is None: continue
        kp_off, v_off, kp_ft, v_ft = predict(s, side)
        gt_kp = np.asarray(s["native_joints_3d"])  # MANO order, 21x3
        gt_v  = s.get("native_vertices")
        # Joint-based PA alignment, applied to vertices (matches viz_seq_mesh)
        _, v_off_a = pa_align_mesh(kp_off, v_off, gt_kp)
        _, v_ft_a  = pa_align_mesh(kp_ft,  v_ft,  gt_kp)
        if gt_v is not None and gt_v.shape == v_off.shape:
            err_off = np.linalg.norm(v_off_a - gt_v, axis=1) * 1000.0
            err_ft  = np.linalg.norm(v_ft_a  - gt_v, axis=1) * 1000.0
            ref = gt_v
        else:
            err_off = np.zeros(778)
            err_ft  = np.linalg.norm(v_ft_a - v_off_a, axis=1) * 1000.0
            ref = v_off_a
        c = ref.mean(axis=0)
        item = {
            "id": f"{kind}_{fid:06d}",
            "frame_id": int(fid),
            "v_off": (v_off_a - c).round(5).tolist(),
            "v_ft":  (v_ft_a  - c).round(5).tolist(),
            "err_off": err_off.round(2).tolist(),
            "err_ft":  err_ft.round(2).tolist(),
        }
        out.append(item)
        print(f"  {kind} {fid}: off mean err {err_off.mean():.1f}, ft {err_ft.mean():.1f}")
    return out


payload = {
    "faces": faces,
    "aria_pr84":   process(aria, "PR84", ARIA_FIDS, "aria_pr84"),
    "pov_scalpel": process(pov,  "R2_s_scalpel_1", POV_FIDS, "pov_scalpel"),
}

with open(OUT, "w") as f:
    json.dump(payload, f)
print(f"wrote {OUT}")
