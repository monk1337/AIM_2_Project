"""Export 100 stratified frames per dataset with 4 models × 2 layer types overlaid.

Two overlay groups per model:
  <Model>-shape : pred_3d (root-aligned) → bbox-anchored weak-perspective. Shows skeleton SHAPE.
  <Model>-pos   : actual model 2D in image frame (perspective_projection through model's own pred_cam).
                  For HaMeR-family: uses scaled_focal + cam_crop_to_full.
                  For MGFM: weak-perspective via pred_camera + bbox transform.
                  For HandOccNet: bbox-anchored fallback (no native 2D output).

GT-HSAM uses stored joints_2d (display-frame).
GT-MPS (Aria only) uses sidecar j2d_v2_disp.
"""
import os
import sys
import json
import argparse
from collections import defaultdict
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from eval_joint_orders import mano_to_op


COLORS = {
    "WiLoR-shape":  "#fca5a5",  # light red
    "WiLoR-pos":    "#dc2626",  # dark red
    "HaMeR-shape":  "#93c5fd",  # light blue
    "HaMeR-pos":    "#1d4ed8",  # dark blue
    "HandOccNet-shape": "#86efac",  # light green
    "HandOccNet-pos":   "#059669",  # dark green
    "MGFM-shape":   "#fcd34d",  # light amber
    "MGFM-pos":     "#b45309",  # dark amber
}

OP_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]


def select_aria(rows, per_pr=20):
    by_pr_frame = defaultdict(set); rows_by_key = defaultdict(list)
    for r in rows:
        by_pr_frame[r["sequence_name"]].add(r["frame_id"])
        rows_by_key[(r["sequence_name"], r["frame_id"])].append(r)
    selected = []
    for pr in sorted(by_pr_frame.keys()):
        frames = sorted(by_pr_frame[pr])
        idxs = np.linspace(0, len(frames) - 1, per_pr).astype(int)
        for i in idxs:
            for r in rows_by_key[(pr, frames[i])]: selected.append(r)
    return selected


def select_pov(rows, per_seq=8):
    by_seq_frame = defaultdict(set); rows_by_key = defaultdict(list)
    for r in rows:
        by_seq_frame[r["sequence_name"]].add(r["frame_id"])
        rows_by_key[(r["sequence_name"], r["frame_id"])].append(r)
    selected = []
    for seq in sorted(by_seq_frame.keys()):
        frames = sorted(by_seq_frame[seq])
        idxs = np.linspace(0, len(frames) - 1, per_seq).astype(int)
        for i in idxs:
            for r in rows_by_key[(seq, frames[i])]: selected.append(r)
    return selected


def project_via_centroid(pred_3d_op, gt_joints_2d, scale_factor=1.4):
    """Anchor pred skeleton at GT centroid (not bbox center, bbox can be clamped if joints exceed image)."""
    cx, cy = float(gt_joints_2d[:, 0].mean()), float(gt_joints_2d[:, 1].mean())
    gt_extent = max(
        float(gt_joints_2d[:, 0].max() - gt_joints_2d[:, 0].min()),
        float(gt_joints_2d[:, 1].max() - gt_joints_2d[:, 1].min()),
    )
    pred_extent = float(np.linalg.norm(pred_3d_op[:, :2], axis=-1).max()) + 1e-6
    s = (gt_extent / scale_factor) / pred_extent
    return np.stack([cx + pred_3d_op[:, 0] * s, cy + pred_3d_op[:, 1] * s], axis=-1).astype(np.float32)


def crop_with_wilor(pipe, image_np, bbox_xywh, is_right):
    from wilor_mini.utils import utils
    from skimage.filters import gaussian
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2.0, y + h / 2.0
    bbox_size = max(w, h)
    flip = (is_right == 0)
    patch_size = pipe.IMAGE_SIZE
    cvimg = image_np.copy()
    ds = (bbox_size / patch_size) / 2.0
    if ds > 1.1:
        cvimg = gaussian(cvimg, sigma=(ds - 1) / 2, channel_axis=2, preserve_range=True)
    img_patch_cv, _ = utils.generate_image_patch_cv2(
        cvimg, cx, cy, bbox_size, bbox_size, patch_size, patch_size,
        flip, 1.0, 0, border_mode=cv2.BORDER_CONSTANT)
    return img_patch_cv, bbox_size, np.array([cx, cy], dtype=np.float32), flip


def predict_wilor(rows, pipe, dtype=torch.float32):
    """Returns dict[key] = (pred_3d_op_root_aligned, bbox, pred_2d_pos_image_frame)."""
    from wilor_mini.utils import utils as wutils
    out_dict = {}
    for r in tqdm(rows, desc="WiLoR"):
        img = np.asarray(get_image(r))
        bbox = derive_bbox_from_joints2d(r["native_joints_2d"], padding=1.5, img_wh=r["image_wh"])
        is_right = 1 if r["hand_side"] == "right" else 0
        patch, bsize, bcenter, flip = crop_with_wilor(pipe, img, bbox, is_right)
        x = torch.from_numpy(patch[None]).to("cuda", dtype=dtype)
        with torch.no_grad():
            o = pipe.wilor_model(x)
        o = {k: v.cpu().float().numpy() for k, v in o.items()}
        # WiLoR pipeline post-processing: cam multiplier first, then 3D flip for left
        pred_cam = o["pred_cam"].copy()
        multiplier = (2 * is_right - 1)
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        pred_3d = o["pred_keypoints_3d"][0].copy()
        if flip:
            pred_3d[:, 0] = -pred_3d[:, 0]
        # 2D in image frame using virtual focal
        img_size_arr = np.asarray(r["image_wh"], dtype=np.float32)
        scaled_focal = pipe.FOCAL_LENGTH / pipe.IMAGE_SIZE * img_size_arr.max()
        pred_cam_t_full = wutils.cam_crop_to_full(
            pred_cam, bcenter[None], bsize, img_size_arr[None], scaled_focal)
        pred_2d_mano = wutils.perspective_projection(
            pred_3d[None], translation=pred_cam_t_full,
            focal_length=np.array([scaled_focal] * 2)[None],
            camera_center=img_size_arr[None] / 2,
        )[0]
        # Convert to OP order
        pred_3d_op = mano_to_op(pred_3d) - mano_to_op(pred_3d)[0:1]
        pred_2d_op = mano_to_op(pred_2d_mano)
        key = (r["sequence_name"], r["frame_id"], r["hand_side"])
        out_dict[key] = (pred_3d_op, bbox, pred_2d_op)
    return out_dict


def predict_hamer(rows, model, cfg):
    out_dict = {}
    img_size = cfg.MODEL.IMAGE_SIZE
    mean = cfg.MODEL.IMAGE_MEAN
    std = cfg.MODEL.IMAGE_STD
    for r in tqdm(rows, desc="HaMeR"):
        img = np.asarray(get_image(r))
        bbox = derive_bbox_from_joints2d(r["native_joints_2d"], padding=1.5, img_wh=r["image_wh"])
        is_right = 1 if r["hand_side"] == "right" else 0
        x_, y_, w_, h_ = bbox
        cx, cy = x_ + w_ / 2.0, y_ + h_ / 2.0
        bsize = max(w_, h_)
        flip = (is_right == 0)
        src = np.array([[cx - bsize / 2, cy - bsize / 2],
                        [cx + bsize / 2, cy - bsize / 2],
                        [cx - bsize / 2, cy + bsize / 2]], dtype=np.float32)
        dst = np.array([[0, 0], [img_size, 0], [0, img_size]], dtype=np.float32)
        M = cv2.getAffineTransform(src, dst)
        crop = cv2.warpAffine(img, M, (img_size, img_size), flags=cv2.INTER_LINEAR)
        if flip: crop = crop[:, ::-1].copy()
        img_t = (crop.astype(np.float32) / 255.0 - np.array(mean)) / np.array(std)
        img_t = np.transpose(img_t, (2, 0, 1)).astype(np.float32)
        x = torch.from_numpy(img_t[None]).cuda()
        with torch.no_grad():
            o = model({"img": x})
        pred_3d = o["pred_keypoints_3d"][0].cpu().float().numpy().copy()
        pred_cam = o["pred_cam"][0:1].cpu().float().numpy().copy()
        if flip:
            pred_3d[:, 0] = -pred_3d[:, 0]
            pred_cam[:, 1] = -pred_cam[:, 1]
        # 2D pos via cam_crop_to_full
        f0 = cfg.EXTRA.FOCAL_LENGTH
        img_arr = np.asarray(r["image_wh"], dtype=np.float32)
        scaled_focal = f0 / img_size * img_arr.max()
        s, tx, ty = pred_cam[0, 0], pred_cam[0, 1], pred_cam[0, 2]
        cx_img, cy_img = img_arr[0] / 2, img_arr[1] / 2
        tz = 2 * scaled_focal / (bsize * s + 1e-9)
        t_x = tx + 2 * (cx - cx_img) / (bsize * s + 1e-9)
        t_y = ty + 2 * (cy - cy_img) / (bsize * s + 1e-9)
        t_full = np.array([[t_x, t_y, tz]])
        pts = pred_3d + t_full
        pred_2d_x = scaled_focal * pts[:, 0] / np.maximum(pts[:, 2], 1e-6) + cx_img
        pred_2d_y = scaled_focal * pts[:, 1] / np.maximum(pts[:, 2], 1e-6) + cy_img
        pred_2d_mano = np.stack([pred_2d_x, pred_2d_y], axis=-1)
        pred_3d_op = mano_to_op(pred_3d) - mano_to_op(pred_3d)[0:1]
        pred_2d_op = mano_to_op(pred_2d_mano)
        key = (r["sequence_name"], r["frame_id"], r["hand_side"])
        out_dict[key] = (pred_3d_op, bbox, pred_2d_op)
    return out_dict


def predict_handoccnet(rows, model, batch_size=64):
    """HandOccNet outputs OP-order 3D camera-frame. -pos layer not generated
    (its 3D coords don't compose cleanly with GT wrist + K projection)."""
    out_dict = {}
    for i in tqdm(range(0, len(rows), batch_size), desc="HandOccNet"):
        chunk = rows[i:i + batch_size]
        patches, meta = [], []
        for r in chunk:
            img = np.asarray(get_image(r))
            bbox = derive_bbox_from_joints2d(r["native_joints_2d"], padding=1.5, img_wh=r["image_wh"])
            is_right = 1 if r["hand_side"] == "right" else 0
            img_used = img if is_right else img[:, ::-1].copy()
            bbox_used = bbox.copy()
            if not is_right:
                bbox_used[0] = img_used.shape[1] - bbox[0] - bbox[2]
            x_, y_, w_, h_ = bbox_used
            cx, cy = x_ + w_ / 2.0, y_ + h_ / 2.0
            bsize = max(w_, h_)
            src = np.array([[cx - bsize / 2, cy - bsize / 2],
                            [cx + bsize / 2, cy - bsize / 2],
                            [cx - bsize / 2, cy + bsize / 2]], dtype=np.float32)
            dst = np.array([[0, 0], [256, 0], [0, 256]], dtype=np.float32)
            M = cv2.getAffineTransform(src, dst)
            crop = cv2.warpAffine(img_used, M, (256, 256), flags=cv2.INTER_LINEAR)
            patch = (crop.astype(np.float32) / 255.0).transpose(2, 0, 1)
            patches.append(patch)
            meta.append({"row": r, "bbox": bbox, "is_right": is_right})
        x = torch.from_numpy(np.stack(patches)).to("cuda")
        with torch.no_grad():
            out = model({"img": x}, {}, {}, "test")
        joints = out["joints_coord_cam"].cpu().float().numpy()  # OP, root-aligned, m
        for j, m in enumerate(meta):
            r = m["row"]
            p3d = joints[j].copy()
            if not m["is_right"]:
                p3d[:, 0] = -p3d[:, 0]
            p3d = p3d - p3d[0:1]
            key = (r["sequence_name"], r["frame_id"], r["hand_side"])
            out_dict[key] = (p3d, m["bbox"], None)  # no pos layer for HandOccNet
    return out_dict


def predict_meshgraphormer(rows, model, mano, mesh_sampler, batch_size=32):
    out_dict = {}
    for i in tqdm(range(0, len(rows), batch_size), desc="MGFM"):
        chunk = rows[i:i + batch_size]
        patches, meta = [], []
        for r in chunk:
            img = np.asarray(get_image(r))
            bbox = derive_bbox_from_joints2d(r["native_joints_2d"], padding=1.5, img_wh=r["image_wh"])
            is_right = 1 if r["hand_side"] == "right" else 0
            img_used = img if is_right else img[:, ::-1].copy()
            bbox_used = bbox.copy()
            if not is_right:
                bbox_used[0] = img_used.shape[1] - bbox[0] - bbox[2]
            x_, y_, w_, h_ = bbox_used
            cx, cy = x_ + w_ / 2.0, y_ + h_ / 2.0
            bsize = max(w_, h_)
            src = np.array([[cx - bsize / 2, cy - bsize / 2],
                            [cx + bsize / 2, cy - bsize / 2],
                            [cx - bsize / 2, cy + bsize / 2]], dtype=np.float32)
            dst = np.array([[0, 0], [224, 0], [0, 224]], dtype=np.float32)
            M = cv2.getAffineTransform(src, dst)
            crop = cv2.warpAffine(img_used, M, (224, 224), flags=cv2.INTER_LINEAR)
            patch = (crop.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            patch = patch.transpose(2, 0, 1).astype(np.float32)
            patches.append(patch)
            meta.append({"row": r, "bbox": bbox, "is_right": is_right})
        x = torch.from_numpy(np.stack(patches)).to("cuda", dtype=torch.float32)
        with torch.no_grad():
            outs = model(x, mano, mesh_sampler)
        _, pred_3d_joints, _, _ = outs[:4]
        pred_3d = pred_3d_joints.cpu().float().numpy()
        for j, m in enumerate(meta):
            r = m["row"]
            p3d = pred_3d[j] - pred_3d[j, 0:1]
            if not m["is_right"]:
                p3d[:, 0] = -p3d[:, 0]
            key = (r["sequence_name"], r["frame_id"], r["hand_side"])
            out_dict[key] = (p3d, m["bbox"], None)  # no pos layer for MGFM
    return out_dict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["aria", "pov", "both"], default="both")
    p.add_argument("--per_pr", type=int, default=20)
    p.add_argument("--per_seq", type=int, default=8)
    p.add_argument("--out_dir", default="/tmp/annotator_export")
    p.add_argument("--max_image_dim", type=int, default=900)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    datasets = []
    if args.dataset in ("aria", "both"): datasets.append("aria")
    if args.dataset in ("pov", "both"): datasets.append("pov")

    print("[setup] Loading WiLoR…")
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    wilor_pipe = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)
    wilor_pipe.wilor_model.eval()

    print("[setup] Loading HaMeR…")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    cwd = os.getcwd()
    os.chdir("/workspace/checkpoints/hamer")
    from hamer.models import load_hamer
    hamer_model, hamer_cfg = load_hamer("/workspace/checkpoints/hamer/checkpoints/hamer.ckpt")
    hamer_model = hamer_model.cuda().eval()
    os.chdir(cwd)

    print("[setup] Loading MeshGraphormer…")
    sys.path.insert(0, "/workspace/code/MeshGraphormer")
    os.chdir("/workspace/code/MeshGraphormer")
    from eval_meshgraphormer import build_model as mgfm_build_model
    mgfm_model, mgfm_mano, mgfm_mesh_sampler = mgfm_build_model()
    os.chdir(cwd)
    for k in list(sys.modules.keys()):
        if k.startswith("manopth") or k.startswith("manolayer"):
            del sys.modules[k]

    print("[setup] Loading HandOccNet…")
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
    ckpt = torch.load("/workspace/checkpoints/handoccnet/HandOccNet_model_dump/snapshot_demo.pth.tar",
                      map_location="cuda", weights_only=False)
    honet_model.load_state_dict(ckpt["network"], strict=False)
    honet_model.eval()

    for ds_name in datasets:
        print(f"\n=== Dataset: {ds_name} ===")
        if ds_name == "aria":
            rows = load_aria_val()
            sel = select_aria(rows, per_pr=args.per_pr)
        else:
            rows = load_pov_test(stride=1)
            sel = select_pov(rows, per_seq=args.per_seq)
        print(f"  {len(sel)} hand-instances")

        wilor_d = predict_wilor(sel, wilor_pipe)
        hamer_d = predict_hamer(sel, hamer_model, hamer_cfg)
        honet_d = predict_handoccnet(sel, honet_model)
        mgfm_d = predict_meshgraphormer(sel, mgfm_model, mgfm_mano, mgfm_mesh_sampler)

        img_dir = f"{args.out_dir}/images/{ds_name}_4models_v2"
        os.makedirs(img_dir, exist_ok=True)
        frames_manifest = []
        by_frame = defaultdict(list)
        for r in sel:
            by_frame[(r["sequence_name"], r["frame_id"])].append(r)

        for order, ((seq, fid), hands) in enumerate(sorted(by_frame.items())):
            img_pil = get_image(hands[0])
            W, H = img_pil.size
            overlays = []
            for r in hands:
                side = "R" if r["hand_side"] == "right" else "L"
                # GT-HSAM
                overlays.append({
                    "name": f"GT-HSAM-{side}",
                    "color": "#a855f7" if side == "R" else "#22d3ee",
                    "keypoints": np.asarray(r["native_joints_2d"]).tolist(),
                    "edges": OP_EDGES,
                })
                # NOTE: GT-MPS overlay dropped, MPS sidecar uses opposite hand_side labels
                # vs HSAM in this dataset, so MPS-R lands on the other hand visually.
                # Metrics still compare correctly via parquet eval_joints_3d (per-row).
                key = (r["sequence_name"], r["frame_id"], r["hand_side"])
                for short, full, dct in [("WiLoR", "WiLoR", wilor_d), ("HaMeR", "HaMeR", hamer_d),
                                          ("HandOccNet", "HandOccNet", honet_d), ("MGFM", "MGFM", mgfm_d)]:
                    if key not in dct: continue
                    p3d, bbox, p2d_pos = dct[key]
                    # Shape layer, use GT joints centroid (handles joints outside image bounds)
                    p2d_shape = project_via_centroid(p3d, np.asarray(r["native_joints_2d"]), scale_factor=1.4)
                    overlays.append({
                        "name": f"{full}-shape-{side}",
                        "color": COLORS[f"{short}-shape"],
                        "keypoints": p2d_shape.tolist(),
                        "edges": OP_EDGES,
                    })
                    # Pos layer (if available)
                    if p2d_pos is not None:
                        overlays.append({
                            "name": f"{full}-pos-{side}",
                            "color": COLORS[f"{short}-pos"],
                            "keypoints": p2d_pos.tolist(),
                            "edges": OP_EDGES,
                        })

            scale = 1.0
            img_save = img_pil
            if max(img_save.size) > args.max_image_dim:
                scale = args.max_image_dim / max(img_save.size)
                img_save = img_save.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
                for ov in overlays:
                    ov["keypoints"] = [[p[0] * scale, p[1] * scale] for p in ov["keypoints"]]

            img_name = f"{seq}_{fid:06d}.jpg"
            img_save.save(f"{img_dir}/{img_name}", quality=85)

            frame_key = f"{seq}/{fid:06d}"
            frames_manifest.append({
                "frameKey": frame_key,
                "imageUrl": f"/images/{ds_name}_4models_v2/{img_name}",
                "orderIndex": order,
                "payload": {"overlays": overlays,
                            "meta": {"seq": seq, "frame_id": int(fid), "scale": scale,
                                     "image_wh_orig": [W, H], "image_wh_disp": list(img_save.size)}}
            })

        manifest = {
            "name": f"{ds_name}_4models_v2",
            "config": {
                "imageWidth": img_save.size[0] if frames_manifest else args.max_image_dim,
                "imageHeight": img_save.size[1] if frames_manifest else args.max_image_dim,
                "description": (
                    f"{ds_name}, 2 layers per model: -shape (light, bbox-anchored) "
                    f"and -pos (dark, actual model 2D in image frame). "
                    f"Aria: -pos for HandOccNet/MGFM is N/A (fisheye, no pinhole projection)."
                )
            },
            "frames": frames_manifest,
        }
        with open(f"{args.out_dir}/manifest_{ds_name}_4models_v2.json", "w") as f:
            json.dump(manifest, f, default=float)
        print(f"  → manifest_{ds_name}_4models_v2.json ({len(frames_manifest)} frames)")
    print("[done]")


if __name__ == "__main__":
    main()
