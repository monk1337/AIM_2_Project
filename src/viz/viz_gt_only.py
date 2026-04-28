"""GT-only visualization: 100 frames per dataset showing just the GT-HSAM joints_2d overlay.

Used to verify that the stored joints_2d (after any necessary CCW90 transform for Aria)
correctly land on the actual hand in the saved image.
"""
import os
import sys
import json
import argparse
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/workspace/code")
from eval_loader import load_aria_val, load_pov_test, get_image


OP_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]


def select_aria(rows, per_pr=20):
    by_pr_frame = defaultdict(set)
    rows_by_key = defaultdict(list)
    for r in rows:
        by_pr_frame[r["sequence_name"]].add(r["frame_id"])
        rows_by_key[(r["sequence_name"], r["frame_id"])].append(r)
    selected = []
    for pr in sorted(by_pr_frame.keys()):
        frames = sorted(by_pr_frame[pr])
        idxs = np.linspace(0, len(frames) - 1, per_pr).astype(int)
        for i in idxs:
            for r in rows_by_key[(pr, frames[i])]:
                selected.append(r)
    return selected


def select_pov(rows, per_seq=8):
    by_seq_frame = defaultdict(set)
    rows_by_key = defaultdict(list)
    for r in rows:
        by_seq_frame[r["sequence_name"]].add(r["frame_id"])
        rows_by_key[(r["sequence_name"], r["frame_id"])].append(r)
    selected = []
    for seq in sorted(by_seq_frame.keys()):
        frames = sorted(by_seq_frame[seq])
        idxs = np.linspace(0, len(frames) - 1, per_seq).astype(int)
        for i in idxs:
            for r in rows_by_key[(seq, frames[i])]:
                selected.append(r)
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["aria", "pov", "both"], default="both")
    p.add_argument("--per_pr", type=int, default=20)
    p.add_argument("--per_seq", type=int, default=8)
    p.add_argument("--out_dir", default="/workspace/annotator_export")
    p.add_argument("--max_image_dim", type=int, default=900)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    datasets = []
    if args.dataset in ("aria", "both"): datasets.append("aria")
    if args.dataset in ("pov", "both"): datasets.append("pov")

    for ds_name in datasets:
        print(f"\n=== Dataset: {ds_name} (GT only) ===")
        if ds_name == "aria":
            rows = load_aria_val()
            sel = select_aria(rows, per_pr=args.per_pr)
        else:
            rows = load_pov_test(stride=1)
            sel = select_pov(rows, per_seq=args.per_seq)
        print(f"  selected {len(sel)} hand-instances")

        img_dir = f"{args.out_dir}/images/{ds_name}_gt"
        os.makedirs(img_dir, exist_ok=True)
        frames_manifest = []
        by_frame = defaultdict(list)
        for r in sel:
            by_frame[(r["sequence_name"], r["frame_id"])].append(r)

        for order, ((seq, fid), hands) in enumerate(tqdm(sorted(by_frame.items()), desc=f"{ds_name} frames")):
            img_pil = get_image(hands[0])
            W, H = img_pil.size
            overlays = []
            for r in hands:
                side_label = "R" if r["hand_side"] == "right" else "L"
                color = "#a855f7" if side_label == "R" else "#22d3ee"  # right=purple, left=cyan
                # also include Aria-MPS GT for Aria val
                if r.get("aria_mps_2d_disp") is not None:
                    overlays.append({
                        "name": f"GT-MPS-{side_label}",
                        "color": "#f59e0b" if side_label == "R" else "#fbbf24",  # amber/yellow
                        "keypoints": np.asarray(r["aria_mps_2d_disp"]).tolist(),
                        "edges": OP_EDGES,
                    })
                overlays.append({
                    "name": f"GT-HSAM-{side_label}",
                    "color": color,
                    "keypoints": np.asarray(r["native_joints_2d"]).tolist(),
                    "edges": OP_EDGES,
                })

            # Resize image + scale overlays
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
                "imageUrl": f"/images/{ds_name}_gt/{img_name}",
                "orderIndex": order,
                "payload": {
                    "overlays": overlays,
                    "meta": {"seq": seq, "frame_id": int(fid), "scale": scale,
                             "image_wh_orig": [W, H], "image_wh_disp": list(img_save.size)}
                }
            })

        manifest = {
            "name": f"{ds_name}_gt",
            "config": {
                "imageWidth": img_save.size[0] if frames_manifest else args.max_image_dim,
                "imageHeight": img_save.size[1] if frames_manifest else args.max_image_dim,
                "description": (
                    f"{ds_name} GT-only viz ({len(frames_manifest)} frames). "
                    f"Aria: HSAM joints_2d (purple/cyan after CCW90) + MPS j2d_v2_disp (amber/yellow). "
                    f"POV: stored synthetic native joints_2d (purple/cyan)."
                )
            },
            "frames": frames_manifest,
        }
        with open(f"{args.out_dir}/manifest_{ds_name}_gt.json", "w") as f:
            json.dump(manifest, f, default=float)
        print(f"  → {len(frames_manifest)} frames → manifest_{ds_name}_gt.json")
    print("[done]")


if __name__ == "__main__":
    main()
