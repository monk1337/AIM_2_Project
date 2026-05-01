"""MeshGraphormer adapter, outputs joints in OP order + 778 vertices, root-aligned, meters.

Uses the official src/modeling/bert/Graphormer_Hand_Network. Weights from HF mirror.
"""
import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import cv2
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

MGFM = "/workspace/code/MeshGraphormer"
sys.path.insert(0, MGFM)
sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, derive_bbox_from_joints2d, get_image
from eval_runner import compute_metrics, run_metrics_aggregation


# Place required model files into MeshGraphormer's expected dirs
def setup_paths():
    os.makedirs(f"{MGFM}/models/graphormer_release", exist_ok=True)
    os.makedirs(f"{MGFM}/models/hrnet", exist_ok=True)
    os.makedirs(f"{MGFM}/src/modeling/data", exist_ok=True)
    src = "/workspace/checkpoints/meshgraphormer/graphormer_hand_state_dict.bin"
    dst = f"{MGFM}/models/graphormer_release/graphormer_hand_state_dict.bin"
    if not os.path.exists(dst): os.symlink(src, dst)
    src = "/workspace/checkpoints/meshgraphormer/hrnetv2_w64_imagenet_pretrained.pth"
    dst = f"{MGFM}/models/hrnet/hrnetv2_w64_imagenet_pretrained.pth"
    if not os.path.exists(dst): os.symlink(src, dst)
    # Need MANO_RIGHT.pkl in src/modeling/data
    dst = f"{MGFM}/src/modeling/data/MANO_RIGHT.pkl"
    if not os.path.exists(dst): os.symlink("/workspace/mano/MANO_RIGHT.pkl", dst)


def build_model():
    setup_paths()
    from src.modeling.bert import BertConfig, Graphormer
    from src.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
    from src.modeling._mano import MANO, Mesh
    from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
    from src.modeling.hrnet.config import config as hrnet_config
    from src.modeling.hrnet.config import update_config as hrnet_update_config

    # Standard config from run_gphmer_handmesh_inference.py
    args = argparse.Namespace(
        num_hidden_layers=4, hidden_size=-1, num_attention_heads=4,
        intermediate_size=-1, input_feat_dim="2051,512,128", hidden_feat_dim="1024,256,64",
        which_gcn="0,0,1", mesh_type="hand", model_name_or_path="src/modeling/bert/bert-base-uncased/",
        config_name="", model_dim_1=512, feedforward_dim_1=2048, model_dim_2=128, feedforward_dim_2=512,
        num_classes=21, drop_out=0.1,
    )

    trans_encoder = []
    input_feat_dim = [int(x) for x in args.input_feat_dim.split(",")]
    hidden_feat_dim = [int(x) for x in args.hidden_feat_dim.split(",")]
    output_feat_dim = input_feat_dim[1:] + [3]

    for i in range(len(output_feat_dim)):
        config_class, model_class = BertConfig, Graphormer
        config = config_class.from_pretrained("bert-base-uncased")
        config.output_attentions = False
        config.hidden_dropout_prob = args.drop_out
        config.img_feature_dim = input_feat_dim[i]
        config.output_feature_dim = output_feat_dim[i]
        args.hidden_size = hidden_feat_dim[i]
        args.intermediate_size = int(args.hidden_size * 2)  # canonical: 2x hidden
        config.graph_conv = bool(int(args.which_gcn.split(",")[i]))
        config.mesh_type = args.mesh_type
        config.num_attention_heads = args.num_attention_heads
        config.num_hidden_layers = args.num_hidden_layers
        config.hidden_size = args.hidden_size
        config.intermediate_size = args.intermediate_size
        config.arch = "hrnet-w64"
        model = model_class(config=config)
        trans_encoder.append(model)
    trans_encoder = torch.nn.Sequential(*trans_encoder)

    # HRNet backbone
    hrnet_yaml = f"{MGFM}/models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
    hrnet_checkpoint = f"{MGFM}/models/hrnet/hrnetv2_w64_imagenet_pretrained.pth"
    if not os.path.exists(hrnet_yaml):
        # download yaml from same source
        os.system(f"hf download hr16/ControlNet-HandRefiner-pruned --include 'cls_hrnet*' --local-dir {MGFM}/models/hrnet 2>&1 | tail -2")
        # fallback: try alternative naming
    hrnet_update_config(hrnet_config, hrnet_yaml)
    backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)

    mano = MANO().cuda(); mano.eval()
    mesh_sampler = Mesh()

    Network = Graphormer_Network(args, config, backbone, trans_encoder)
    sd = torch.load("/workspace/checkpoints/meshgraphormer/graphormer_hand_state_dict.bin",
                    map_location="cuda", weights_only=False)
    Network.load_state_dict(sd, strict=False)
    Network = Network.cuda().eval()
    return Network, mano, mesh_sampler


def mgfm_crop(img_np, bbox_xywh):
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2.0, y + h / 2.0
    bbox_size = max(w, h)
    src_pts = np.array([[cx - bbox_size / 2, cy - bbox_size / 2],
                        [cx + bbox_size / 2, cy - bbox_size / 2],
                        [cx - bbox_size / 2, cy + bbox_size / 2]], dtype=np.float32)
    dst_pts = np.array([[0, 0], [224, 0], [0, 224]], dtype=np.float32)
    M = cv2.getAffineTransform(src_pts, dst_pts)
    crop = cv2.warpAffine(img_np, M, (224, 224), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    img_t = crop.astype(np.float32) / 255.0
    img_t = (img_t - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_t = np.transpose(img_t, (2, 0, 1)).astype(np.float32)
    return img_t, bbox_size, np.array([cx, cy], dtype=np.float32)


def run_mgfm(rows, model, mano, mesh_sampler, batch_size=32):
    print("[crop] Building patches...")
    patches, meta = [], []
    for r in tqdm(rows, desc="crop"):
        img = np.asarray(get_image(r))
        bbox = derive_bbox_from_joints2d(r["native_joints_2d"], padding=1.5, img_wh=r["image_wh"])
        is_right = 1 if r["hand_side"] == "right" else 0
        if not is_right:
            img = img[:, ::-1].copy()
            bbox = bbox.copy()
            bbox[0] = img.shape[1] - bbox[0] - bbox[2]
        patch, bsize, bcenter = mgfm_crop(img, bbox)
        patches.append(patch)
        meta.append({"row": r, "bbox_size": bsize, "box_center": bcenter, "is_right": is_right,
                     "img_wh": r["image_wh"], "flip": (not is_right)})

    print("[fwd] Forward...")
    samples = []
    bs = batch_size
    for i in tqdm(range(0, len(patches), bs), desc="forward"):
        batch_patches = np.stack(patches[i:i + bs])
        batch_meta = meta[i:i + bs]
        x = torch.from_numpy(batch_patches).to("cuda", dtype=torch.float32)
        with torch.no_grad():
            outs = model(x, mano, mesh_sampler)
        # Outputs vary by version: at minimum (pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices)
        pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices = outs[:4]
        pj_np = pred_3d_joints.cpu().float().numpy()    # (B, 21, 3) OP order
        pv_np = pred_vertices.cpu().float().numpy()      # (B, 778, 3)
        # Root-align (subtract wrist)
        for j, m in enumerate(batch_meta):
            r = m["row"]
            p3d = pj_np[j].copy()
            pverts = pv_np[j].copy()
            wrist = p3d[0:1]  # OP[0] = Wrist
            p3d = p3d - wrist
            pverts = pverts - wrist
            if m["flip"]:
                p3d[:, 0] = -p3d[:, 0]
                pverts[:, 0] = -pverts[:, 0]
            samples.append({
                "row": r,
                "pred_3d_mano": p3d,    # actually OP order (pred_in_op_order=True downstream)
                "pred_2d_mano": np.zeros((21, 2), dtype=np.float32),
                "pred_verts_mano": pverts,
                "pred_cam_t_full": np.zeros(3, dtype=np.float32),
            })
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["aria_val", "pov_test"], default="pov_test")
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()
    if args.out is None:
        suffix = f"_stride{args.stride}" if args.dataset == "pov_test" and args.stride > 1 else ""
        args.out = f"/workspace/results/meshgraphormer_{args.dataset}{suffix}.json"

    print(f"[1/4] Loading {args.dataset}...")
    rows = load_aria_val() if args.dataset == "aria_val" else load_pov_test(stride=args.stride)
    if args.limit:
        rows = rows[:args.limit]
    print(f"  {len(rows)} instances")

    print("[2/4] Loading MeshGraphormer...")
    model, mano, mesh_sampler = build_model()

    t0 = time.time()
    samples = run_mgfm(rows, model, mano, mesh_sampler, batch_size=args.batch_size)
    elapsed = time.time() - t0

    print(f"[3/4] Computing metrics ({args.dataset})...")
    samples = compute_metrics(samples, args.dataset, root_aligned_model=True, pred_in_op_order=True)

    print("[4/4] Aggregating...")
    agg, per_sample = run_metrics_aggregation(samples, args.dataset, "MeshGraphormer (off-the-shelf)", True)
    agg["elapsed_sec"] = elapsed

    print("\n=== OVERALL ===")
    keys = ["aria_native_pa_mpjpe_mm", "hsam_pve_mm", "n"] if args.dataset == "aria_val" else \
           ["pov_native_mpjpe_mm", "pov_native_pa_mpjpe_mm", "pov_native_pve_mm", "pov_native_pa_pve_mm", "n"]
    for k in keys:
        v = agg.get(k)
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== PER-VIDEO ===")
    pa_key = "aria_native_pa_mpjpe_mm" if args.dataset == "aria_val" else "pov_native_pa_mpjpe_mm"
    for seq, g in sorted(agg["per_group"].items()):
        print(f"  {seq}: PA={g.get(pa_key, float('nan')):.2f} mm  n={g['n']}")

    with open(args.out, "w") as f:
        json.dump({"summary": agg, "per_sample": per_sample}, f, default=float)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
