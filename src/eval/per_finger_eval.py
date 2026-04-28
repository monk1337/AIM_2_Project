"""Compute per-finger PA-MPJPE for off-shelf and FT WiLoR on POV-Surgery and Aria.

OpenPose-21 finger joint indices (excluding wrist):
  thumb:  [1, 2, 3, 4]
  index:  [5, 6, 7, 8]
  middle: [9, 10, 11, 12]
  ring:   [13, 14, 15, 16]
  pinky:  [17, 18, 19, 20]

Output JSON: /workspace/results/per_finger.json
"""
import sys, json, numpy as np, torch
sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from viz_seq_mesh import crop_for_wilor
from eval_metrics import procrustes_align
from eval_joint_orders import MANO_TO_OPENPOSE
from torch.amp import autocast

FINGERS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}
OUT = "/workspace/results/per_finger.json"

print("[1/5] loading models")
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
pipe_off = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
m_off = pipe_off.wilor_model.eval()
pipe_ft = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
ck = torch.load("/workspace/checkpoints/wilor_ft_mixed/wilor_ft_final.pth",
                map_location="cuda", weights_only=False)
pipe_ft.wilor_model.load_state_dict(ck["model_state_dict"])
m_ft = pipe_ft.wilor_model.eval()


def predict(samp, side, model):
    img = np.asarray(get_image(samp))
    bbox = derive_bbox_from_joints2d(samp["native_joints_2d"], padding=1.5,
                                     img_wh=samp["image_wh"])
    crop_bgr_rgb, flip, _ = crop_for_wilor(img, bbox, side)
    x = torch.from_numpy(crop_bgr_rgb[:, :, ::-1].astype(np.float32)[None]).cuda().float()
    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
        r = model(x)
    kp = r["pred_keypoints_3d"][0].float().cpu().numpy()
    if flip:
        kp = kp.copy(); kp[:, 0] = -kp[:, 0]
    return kp[MANO_TO_OPENPOSE]   # OP-21 order


def per_joint_err(pred_op, gt_op):
    pred_a, _, _, _ = procrustes_align(pred_op, gt_op)
    return np.linalg.norm(pred_a - gt_op, axis=1) * 1000.0  # mm


def aggregate(samples, label, max_n=600):
    print(f"[run] {label} ({len(samples)} samples, eval up to {max_n})")
    rng = np.random.default_rng(0)
    if len(samples) > max_n:
        ix = rng.choice(len(samples), size=max_n, replace=False)
        samples = [samples[i] for i in sorted(ix)]
    err_off = np.zeros((len(samples), 21))
    err_ft  = np.zeros((len(samples), 21))
    valid   = np.zeros(len(samples), dtype=bool)
    for i, s in enumerate(samples):
        try:
            side = s["hand_side"]
            gt_op = np.asarray(s["native_joints_3d"])[MANO_TO_OPENPOSE]
            kp_off = predict(s, side, m_off)
            kp_ft  = predict(s, side, m_ft)
            err_off[i] = per_joint_err(kp_off, gt_op)
            err_ft[i]  = per_joint_err(kp_ft,  gt_op)
            valid[i]   = True
        except Exception as e:
            print(f"  skip {i}: {e}")
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(samples)}")
    err_off = err_off[valid]
    err_ft  = err_ft[valid]
    by_finger = {}
    for name, idx in FINGERS.items():
        by_finger[name] = {
            "off_mean": float(err_off[:, idx].mean()),
            "off_std":  float(err_off[:, idx].std()),
            "ft_mean":  float(err_ft[:, idx].mean()),
            "ft_std":   float(err_ft[:, idx].std()),
        }
    return {"n": int(valid.sum()), "by_finger": by_finger}


print("[2/5] loading POV-Surgery test")
pov = load_pov_test(stride=10)
print("[3/5] loading Aria val (PR84 + filtered)")
aria = [s for s in load_aria_val() if s.get("sequence_name") in ("PR81", "PR82", "PR84")]

result = {
    "pov":  aggregate(pov,  "POV-Surgery"),
    "aria": aggregate(aria, "Aria HSAM"),
}

with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(f"[5/5] wrote {OUT}")
print(json.dumps(result, indent=2))
