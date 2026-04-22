import os
import pickle
import random
from collections import defaultdict

import cv2
import numpy as np
from datasets import Dataset
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_OOD = os.path.dirname(_HERE)
SRC = os.path.join(_OOD, "aria_val_raw", "aria_val_pr83_pr84_clean_20260420")
DST = os.path.join(_OOD, "common_format", "aria")
IMG_DIR = os.path.join(DST, "images")
OV_DIR = os.path.join(DST, "sanity_overlays")


def bbox_from_kp2d(kp, W, H, pad=1.5):
    lo = kp.min(0)
    hi = kp.max(0)
    center = (lo + hi) / 2
    cx, cy = float(center[0]), float(center[1])
    size = float((hi - lo).max() * pad)
    x1 = max(0.0, cx - size / 2)
    y1 = max(0.0, cy - size / 2)
    x2 = min(float(W), cx + size / 2)
    y2 = min(float(H), cy + size / 2)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(OV_DIR, exist_ok=True)

    ds = Dataset.load_from_disk(SRC)
    samples = []
    saved_images = set()
    skipped = []
    bbox_ws = []
    bbox_hs = []
    per_side = defaultdict(int)
    per_seq = defaultdict(int)
    seq_side_indices = defaultdict(list)

    for i, row in enumerate(ds):
        seq = row["sequence_name"]
        fid = int(row["frame_id"])
        side = row["hand_side"]
        img = row["image"].transpose(Image.ROTATE_270)
        W, H = img.size
        K = np.array(row["intrinsics"], dtype=np.float32).reshape(3, 3)
        j2d = np.array(row["joints_2d"], dtype=np.float32).reshape(21, 2)
        j3d = np.array(row["eval_joints_3d"], dtype=np.float32).reshape(21, 3)

        if np.any(~np.isfinite(j2d)) or np.any(~np.isfinite(j3d)):
            skipped.append((seq, fid, side, "nan"))
            continue
        if j2d[:, 0].max() < 0 or j2d[:, 1].max() < 0 or j2d[:, 0].min() >= W or j2d[:, 1].min() >= H:
            skipped.append((seq, fid, side, "oob"))
            continue

        img_rel = f"images/{seq}_{fid:06d}.png"
        img_abs = os.path.join(DST, img_rel)
        if img_abs not in saved_images:
            img.save(img_abs, format="PNG")
            saved_images.add(img_abs)

        bbox = bbox_from_kp2d(j2d, W, H)
        bbox_ws.append(bbox[2] - bbox[0])
        bbox_hs.append(bbox[3] - bbox[1])

        sample = {
            "frame_id": f"{seq}/{fid:06d}/{side}",
            "image_path": img_rel,
            "K": K,
            "joints_3d": j3d,
            "joints_3d_frame_trustworthy": False,
            "vertices_3d": None,
            "joints_2d": j2d,
            "bbox": bbox,
            "is_right": side == "right",
            "hand_side": side,
            "sequence_name": seq,
            "frame_id_int": fid,
        }
        samples.append(sample)
        per_side[side] += 1
        per_seq[seq] += 1
        seq_side_indices[side].append(len(samples) - 1)

    with open(os.path.join(DST, "samples.pkl"), "wb") as f:
        pickle.dump(samples, f)

    rng = random.Random(42)
    picks = []
    for side in ("right", "left"):
        idxs = seq_side_indices[side]
        rng.shuffle(idxs)
        picks.extend(idxs[:3])

    for idx in picks:
        s = samples[idx]
        img_abs = os.path.join(DST, s["image_path"])
        im = cv2.imread(img_abs)
        for (x, y) in s["joints_2d"]:
            cv2.circle(im, (int(round(x)), int(round(y))), 4, (0, 255, 0), -1)
        x1, y1, x2, y2 = s["bbox"].astype(int)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"Aria {s['frame_id']} {s['hand_side']}"
        cv2.putText(im, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        out = os.path.join(OV_DIR, f"{s['sequence_name']}_{s['frame_id_int']:06d}_{s['hand_side']}.png")
        cv2.imwrite(out, im)

    print("=" * 60)
    print(f"samples written : {len(samples)}")
    print(f"unique images   : {len(saved_images)}")
    print(f"by side         : {dict(per_side)}")
    print(f"by sequence     : {dict(per_seq)}")
    if bbox_ws:
        print(f"bbox mean (w,h) : ({np.mean(bbox_ws):.1f}, {np.mean(bbox_hs):.1f})")
        print(f"bbox min/max w  : {np.min(bbox_ws):.1f} / {np.max(bbox_ws):.1f}")
        print(f"bbox min/max h  : {np.min(bbox_hs):.1f} / {np.max(bbox_hs):.1f}")
    print(f"skipped         : {len(skipped)}")
    for s in skipped[:20]:
        print("  ", s)
    print(f"overlays        : {len(picks)} in {OV_DIR}")


if __name__ == "__main__":
    main()
