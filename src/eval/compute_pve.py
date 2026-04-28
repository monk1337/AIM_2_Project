"""Quick: compute PVE for the 12 selected frames and dump to JSON."""
import sys, json, numpy as np, torch
sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from viz_seq_mesh import crop_for_wilor
from eval_metrics import procrustes_align
from torch.amp import autocast

ARIA_FIDS = [104, 164, 245, 287, 377, 415]
POV_FIDS  = [179, 357, 535, 713, 891, 1959]

print("loading models")
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
pipe_off = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
m_off = pipe_off.wilor_model.eval()
pipe_ft = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
ck = torch.load("/workspace/checkpoints/wilor_ft_anchored/wilor_ft_final.pth", map_location="cuda", weights_only=False)
pipe_ft.wilor_model.load_state_dict(ck["model_state_dict"])
m_ft = pipe_ft.wilor_model.eval()

print("loading data")
aria = {(s["sequence_name"], int(s["frame_id"]), s["hand_side"]): s for s in load_aria_val() if s["sequence_name"] == "PR84"}
pov = {(s["sequence_name"], int(s["frame_id"]), s["hand_side"]): s for s in load_pov_test(stride=1) if s["sequence_name"] == "R2_s_scalpel_1"}

results = []

def run(samp, kind, side):
    img = np.asarray(get_image(samp))
    bbox = derive_bbox_from_joints2d(samp["native_joints_2d"], padding=1.5, img_wh=samp["image_wh"])
    crop_bgr_rgb, flip, M = crop_for_wilor(img, bbox, side)
    x = torch.from_numpy(crop_bgr_rgb[:, :, ::-1].astype(np.float32)[None]).cuda().float()
    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
        r_off = m_off(x); r_ft = m_ft(x)
    v_off = r_off["pred_vertices"][0].float().cpu().numpy()
    v_ft  = r_ft["pred_vertices"][0].float().cpu().numpy()
    if flip:
        v_off = v_off.copy(); v_off[:, 0] = -v_off[:, 0]
        v_ft  = v_ft.copy();  v_ft[:, 0]  = -v_ft[:, 0]
    gt_v = samp.get("native_vertices")
    if gt_v is None or gt_v.shape != v_off.shape:
        return None
    v_off_a = procrustes_align(v_off, gt_v)[0]
    v_ft_a  = procrustes_align(v_ft,  gt_v)[0]
    pve_off = float(np.linalg.norm(v_off_a - gt_v, axis=1).mean()) * 1000.0
    pve_ft  = float(np.linalg.norm(v_ft_a  - gt_v, axis=1).mean()) * 1000.0
    return pve_off, pve_ft

# helper to find sample regardless of side
def pick(seq_dict, seq, fid):
    for k, s in seq_dict.items():
        if k[0] == seq and k[1] == fid:
            return s, k[2]
    return None, None

for fid in ARIA_FIDS:
    s, side = pick(aria, "PR84", fid)
    if s is None: continue
    out = run(s, "aria", side)
    if out: results.append({"seq": "aria_PR84", "fid": fid, "pve_off": round(out[0],1), "pve_ft": round(out[1],1)})
    print(f"  aria PR84 {fid} side={side} -> {out}")

for fid in POV_FIDS:
    s, side = pick(pov, "R2_s_scalpel_1", fid)
    if s is None: continue
    out = run(s, "pov", side)
    if out: results.append({"seq": "pov_R2_s_scalpel_1", "fid": fid, "pve_off": round(out[0],1), "pve_ft": round(out[1],1)})
    print(f"  pov scalpel {fid} side={side} -> {out}")

with open("/workspace/results/pve_extra.json", "w") as f:
    json.dump(results, f, indent=2)
print("done", len(results))
