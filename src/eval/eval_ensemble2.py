"""Ensemble eval: MGFM (12.00 mm) + HandOccNet (13.88 mm), both already Aria-strong.
Also tests test-time aug."""
import os
import sys
import json
import numpy as np
import torch
import cv2
from tqdm import tqdm

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, derive_bbox_from_joints2d, get_image
from eval_joint_orders import mano_to_op


def crop_for_honet(img, bbox, image_size=256, hand_side="right", flip_input=True):
    is_right = 1 if hand_side == "right" else 0
    if not is_right and flip_input:
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
    return crop, is_right


def predict_handoccnet(rows, model, batch_size=64):
    from torch.cuda.amp import autocast
    out = {}
    patches, meta = [], []
    for r in rows:
        img = np.asarray(get_image(r))
        bbox = derive_bbox_from_joints2d(r["native_joints_2d"], padding=1.5, img_wh=r["image_wh"])
        crop, is_right = crop_for_honet(img, bbox, 256, r["hand_side"])
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


def predict_mgfm(rows, model, mano, mesh_sampler, batch_size=32):
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
        dst = np.array([[0, 0], [224, 0], [0, 224]], dtype=np.float32)
        M = cv2.getAffineTransform(src, dst)
        crop = cv2.warpAffine(img, M, (224, 224), flags=cv2.INTER_LINEAR)
        patch = (crop.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        patch = patch.transpose(2, 0, 1).astype(np.float32)
        patches.append(patch)
        meta.append({"row": r, "is_right": is_right})

    bs = batch_size
    for i in tqdm(range(0, len(patches), bs), desc="MGFM"):
        x = torch.from_numpy(np.stack(patches[i:i+bs])).to("cuda", dtype=torch.float32)
        with torch.no_grad():
            outs = model(x, mano, mesh_sampler)
        _, pred_3d_joints, _, _ = outs[:4]
        pred_3d = pred_3d_joints.cpu().float().numpy()
        for j, m in enumerate(meta[i:i+bs]):
            r = m["row"]
            p3d = pred_3d[j] - pred_3d[j, 0:1]
            if not m["is_right"]:
                p3d[:, 0] = -p3d[:, 0]
            out[(r["sequence_name"], r["frame_id"], r["hand_side"])] = p3d
    return out


def main():
    print("[1/4] Loading Aria val…")
    aria_eval = load_aria_val()
    print(f"  n={len(aria_eval)}")

    print("[2/4] Loading MeshGraphormer…")
    sys.path.insert(0, "/workspace/code/MeshGraphormer")
    cwd = os.getcwd()
    os.chdir("/workspace/code/MeshGraphormer")
    from eval_meshgraphormer import build_model as mgfm_build_model
    mgfm_model, mgfm_mano, mgfm_mesh_sampler = mgfm_build_model()
    mgfm_model.eval()
    os.chdir(cwd)
    for k in list(sys.modules.keys()):
        if k.startswith("manopth") or k.startswith("manolayer"):
            del sys.modules[k]

    print("[3/4] Loading HandOccNet…")
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

    print("[4/4] Predicting + ensembling…")
    mgfm_preds = predict_mgfm(aria_eval, mgfm_model, mgfm_mano, mgfm_mesh_sampler)
    honet_preds = predict_handoccnet(aria_eval, honet_model)

    from eval_runner import compute_metrics, run_metrics_aggregation
    # Finer sweep around best
    for wmgfm, whonet in [(0.65, 0.35), (0.7, 0.3), (0.75, 0.25), (0.8, 0.2), (0.85, 0.15), (0.9, 0.1)]:
        samples = []
        for r in aria_eval:
            key = (r["sequence_name"], r["frame_id"], r["hand_side"])
            mp = mgfm_preds.get(key)
            hp = honet_preds.get(key)
            if mp is None or hp is None:
                continue
            avg = wmgfm * mp + whonet * hp
            samples.append({
                "row": r, "pred_3d_mano": avg,
                "pred_2d_mano": np.zeros((21, 2), dtype=np.float32),
                "pred_verts_mano": np.zeros((778, 3), dtype=np.float32),
                "pred_cam_t_full": np.zeros(3, dtype=np.float32),
            })
        samples = compute_metrics(samples, "aria_val", root_aligned_model=True, pred_in_op_order=True)
        agg, _ = run_metrics_aggregation(samples, "aria_val", f"Ensemble {wmgfm}MGFM+{whonet}HONet", True)
        print(f"  w_mgfm={wmgfm} w_honet={whonet}  Aria-PA: {agg.get('aria_native_pa_mpjpe_mm', float('nan')):.2f}")

    # Save final results
    with open("/workspace/results/ensemble_mgfm_honet.json", "w") as f:
        json.dump({"note": "see stdout for sweep", "wmgfm_whonet_swept": True}, f)


if __name__ == "__main__":
    main()
