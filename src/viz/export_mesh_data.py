"""Export 3D mesh data (vertices for off+ft, faces, per-vertex error) for the
12 selected dashboard frames into a single JSON. Faces are shared (MANO topology),
vertices are PA-aligned to GT for fair morph animation.
"""
import sys, json, numpy as np, torch
sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from viz_seq_mesh import crop_for_wilor
from eval_metrics import procrustes_align
from torch.amp import autocast

ARIA_FIDS = [104, 164, 245, 287, 377, 415]
POV_FIDS  = [179, 357, 535, 713, 891, 1959]
OUT = "/workspace/results/mesh_export.json"

print("loading models")
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
pipe_off = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
m_off = pipe_off.wilor_model.eval()
pipe_ft = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
ck = torch.load("/workspace/checkpoints/wilor_ft_anchored/wilor_ft_final.pth",
                map_location="cuda", weights_only=False)
pipe_ft.wilor_model.load_state_dict(ck["model_state_dict"])
m_ft = pipe_ft.wilor_model.eval()

# MANO faces (1538 triangles): grab them once from the pipeline
faces = pipe_off.wilor_model.mano.faces.astype(np.int32).tolist()
print(f"faces: {len(faces)} triangles")

aria = {(s["sequence_name"], int(s["frame_id"]), s["hand_side"]): s
        for s in load_aria_val() if s["sequence_name"] == "PR84"}
pov = {(s["sequence_name"], int(s["frame_id"]), s["hand_side"]): s
       for s in load_pov_test(stride=1) if s["sequence_name"] == "R2_s_scalpel_1"}


def pick(seq_dict, seq, fid):
    for k, s in seq_dict.items():
        if k[0] == seq and k[1] == fid:
            return s, k[2]
    return None, None


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
    if flip:
        v_off = v_off.copy(); v_off[:, 0] = -v_off[:, 0]
        v_ft  = v_ft.copy();  v_ft[:, 0]  = -v_ft[:, 0]
    return v_off, v_ft


def process(seq_dict, seq_key, fids, kind):
    out = []
    for fid in fids:
        s, side = pick(seq_dict, seq_key, fid)
        if s is None: continue
        v_off, v_ft = predict(s, side)
        gt_v = s.get("native_vertices")
        # PA-align both predictions to GT (or just to off if no GT) so they
        # share a coordinate frame for morphing.
        if gt_v is not None and gt_v.shape == v_off.shape:
            v_off_a = procrustes_align(v_off, gt_v)[0]
            v_ft_a  = procrustes_align(v_ft,  gt_v)[0]
            err_off = np.linalg.norm(v_off_a - gt_v, axis=1) * 1000.0
            err_ft  = np.linalg.norm(v_ft_a  - gt_v, axis=1) * 1000.0
            ref = gt_v
        else:
            v_ft_a = procrustes_align(v_ft, v_off)[0]
            v_off_a = v_off
            err_off = np.zeros(778)
            err_ft  = np.linalg.norm(v_ft_a - v_off_a, axis=1) * 1000.0
            ref = v_off_a
        # Center for nicer rendering
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
print(f"wrote {OUT} ({sum(len(payload[k]) for k in ['aria_pr84','pov_scalpel'])} frames)")
