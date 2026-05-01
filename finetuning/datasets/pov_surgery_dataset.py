"""
POV-Surgery dataset for WiLoR finetuning.
Outputs batches matching WiLoR's compute_loss() expectations.
"""

import os
import sys
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.filters import gaussian

# Add WiLoR to path for dataset utils
WILOR_ROOT = Path(__file__).resolve().parents[3] / "WiLoR"
sys.path.insert(0, str(WILOR_ROOT))

from wilor.datasets.utils import (
    expand_to_aspect_ratio,
    gen_trans_from_patch_cv,
    convert_cvimg_to_tensor,
    trans_point2d,
)

# Add our utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.gt_processing import (
    MANO_TO_OPENPOSE,
    build_gt_mano, mano_forward, transform_to_camera,
    compute_global_orient_camera_frame, load_gt_annotation,
    derive_bbox_from_2d_joints,
)

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])

# Target aspect ratio for ViT backbone
BBOX_SHAPE = [192, 256]


def _generate_patch_and_transform(img_bgr, center_x, center_y, bbox_size,
                                   patch_size, scale, rot):
    """Generate image patch and affine transform matrix.

    Args:
        img_bgr: (H, W, 3) BGR image
        center_x, center_y: bbox center in original image
        bbox_size: square bbox side length
        patch_size: output patch size (256)
        scale: augmentation scale factor
        rot: augmentation rotation (degrees)

    Returns:
        img_patch_rgb: (patch_size, patch_size, 3) RGB float32
        trans: (2, 3) affine transform matrix (original pixels -> crop pixels)
    """
    trans = gen_trans_from_patch_cv(
        center_x, center_y, bbox_size, bbox_size,
        patch_size, patch_size, scale, rot,
    )
    img_patch = cv2.warpAffine(
        img_bgr, trans, (patch_size, patch_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    # BGR -> RGB
    img_patch_rgb = img_patch[:, :, ::-1].copy().astype(np.float32)
    return img_patch_rgb, trans


class POVSurgeryDataset(Dataset):
    """POV-Surgery dataset for WiLoR finetuning.

    Produces batches matching WiLoR's compute_loss() format:
        img:              [3, 256, 256] ImageNet-normalized
        keypoints_2d:     [21, 3]  (x, y in [-0.5, 0.5], confidence)
        keypoints_3d:     [21, 4]  (x, y, z in meters, confidence)
        mano_params:      dict with global_orient [3], hand_pose [45], betas [10]
        has_mano_params:  dict with binary flags
        mano_params_is_axis_angle: dict with True/False
    """

    def __init__(
        self,
        data_root,
        mano_model_dir,
        split="train",
        img_size=256,
        rescale_factor=2.5,
        pad_factor=1.5,
        augment=True,
        aug_config=None,
        cache_dir=None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.rescale_factor = rescale_factor
        self.pad_factor = pad_factor
        self.augment = augment
        self.split = split

        # Augmentation config
        if aug_config is None:
            aug_config = {}
        self.aug_cfg = {
            "SCALE_FACTOR": aug_config.get("SCALE_FACTOR", 0.3),
            "ROT_FACTOR": aug_config.get("ROT_FACTOR", 30),
            "ROT_AUG_RATE": aug_config.get("ROT_AUG_RATE", 0.6),
            "TRANS_FACTOR": aug_config.get("TRANS_FACTOR", 0.02),
            "COLOR_SCALE": aug_config.get("COLOR_SCALE", 0.2),
        }

        # Load split info
        if split == "train":
            split_pkl = self.data_root / "handoccnet_train" / "2d_repro_ho3d_style_hocc_cleaned.pkl"
        else:
            split_pkl = self.data_root / "handoccnet_train" / "2d_repro_ho3d_style_test_cleaned.pkl"

        with open(split_pkl, "rb") as f:
            split_info = pickle.load(f)

        # Build sample list
        self.samples = []
        for pkl_key, val in split_info.items():
            seq, frame_id = pkl_key.split("/")
            pkl_path = self.data_root / "annotation" / seq / f"{frame_id}.pkl"
            img_path = self.data_root / "color" / seq / f"{frame_id}.jpg"
            if pkl_path.exists() and img_path.exists():
                # Swap x<->y in joints_uv (HO3D convention -> pixel x,y)
                juv_raw = val["joints_uv"]
                juv = np.zeros_like(juv_raw, dtype=np.float32)
                juv[:, 0] = juv_raw[:, 1]
                juv[:, 1] = juv_raw[:, 0]
                # Reorder MANO -> OpenPose
                juv = juv[MANO_TO_OPENPOSE]
                self.samples.append((pkl_key, str(pkl_path), str(img_path), juv))

        print(f"[POVSurgeryDataset] split={split}, samples={len(self.samples)}")

        # Precompute GT cache
        if cache_dir is None:
            cache_dir = self.data_root / "gt_cache_wilor_ft"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"gt_cache_{split}.pkl"

        if cache_path.exists():
            print(f"  Loading GT cache from {cache_path}...")
            with open(cache_path, "rb") as f:
                self.gt_cache = pickle.load(f)
            print(f"  Loaded {len(self.gt_cache)} cached entries.")
        else:
            print(f"  Precomputing GT for {len(self.samples)} samples...")
            self.gt_cache = self._precompute_gt(mano_model_dir)
            with open(cache_path, "wb") as f:
                pickle.dump(self.gt_cache, f)
            print(f"  Saved GT cache to {cache_path}")

    def _precompute_gt(self, mano_model_dir):
        """Precompute GT 3D joints, vertices, and camera-frame MANO params."""
        device = "cpu"
        gt_mano, tip_ids = build_gt_mano(mano_model_dir, device)

        gt_cache = {}
        for i, (key, pkl_path, img_path, joints_uv) in enumerate(self.samples):
            if (i + 1) % 2000 == 0:
                print(f"    [{i+1}/{len(self.samples)}]")

            anno = load_gt_annotation(pkl_path)
            joints_local, verts_local = mano_forward(
                gt_mano, tip_ids,
                anno["global_orient"], anno["hand_pose"], anno["betas"],
                device,
            )
            joints_cam, verts_cam = transform_to_camera(joints_local, verts_local, anno)
            global_orient_cam = compute_global_orient_camera_frame(anno["global_orient"], anno)

            gt_cache[key] = {
                "joints_cam": joints_cam.astype(np.float32),
                "verts_cam": verts_cam.astype(np.float32),
                "global_orient_cam": global_orient_cam,
                "hand_pose": anno["hand_pose"].flatten().astype(np.float32),
                "betas": anno["betas"].flatten().astype(np.float32),
            }

        return gt_cache

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, pkl_path, img_path, joints_2d = self.samples[idx]
        gt = self.gt_cache[key]

        # ── Load image (BGR, cv2 native) ──
        img_bgr = cv2.imread(img_path)

        # ── Compute bbox and center/scale ──
        joints_2d = joints_2d.copy()  # (21, 2) pixel coords, OpenPose order
        bbox = derive_bbox_from_2d_joints(joints_2d, img_shape=img_bgr.shape[:2], pad_factor=self.pad_factor)

        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        scale = self.rescale_factor * (bbox[2:4] - bbox[0:2]) / 200.0  # (2,)

        # Expand to ViT aspect ratio
        bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=BBOX_SHAPE).max()

        # ── Augmentation ──
        aug_scale = 1.0
        aug_rot = 0.0
        color_scale = [1.0, 1.0, 1.0]

        if self.augment:
            aug_scale = np.clip(np.random.randn(), -1.0, 1.0) * self.aug_cfg["SCALE_FACTOR"] + 1.0
            if random.random() <= self.aug_cfg["ROT_AUG_RATE"]:
                aug_rot = np.clip(np.random.randn(), -2.0, 2.0) * self.aug_cfg["ROT_FACTOR"]
            tx = np.clip(np.random.randn(), -1.0, 1.0) * self.aug_cfg["TRANS_FACTOR"]
            ty = np.clip(np.random.randn(), -1.0, 1.0) * self.aug_cfg["TRANS_FACTOR"]
            center_x += tx * bbox_size
            center_y += ty * bbox_size
            c_up = 1.0 + self.aug_cfg["COLOR_SCALE"]
            c_low = 1.0 - self.aug_cfg["COLOR_SCALE"]
            color_scale = [random.uniform(c_low, c_up) for _ in range(3)]

        # ── Anti-aliasing blur ──
        downsampling_factor = (bbox_size * aug_scale) / self.img_size / 2.0
        if downsampling_factor > 1.1:
            img_bgr = gaussian(img_bgr, sigma=(downsampling_factor - 1) / 2,
                               channel_axis=2, preserve_range=True).astype(np.uint8)

        # ── Generate image patch ──
        img_patch_rgb, trans = _generate_patch_and_transform(
            img_bgr, center_x, center_y, bbox_size,
            self.img_size, aug_scale, aug_rot,
        )

        # Color augmentation
        for c in range(3):
            img_patch_rgb[:, :, c] = np.clip(img_patch_rgb[:, :, c] * color_scale[c], 0, 255)

        # HWC -> CHW tensor, normalize
        img_tensor = convert_cvimg_to_tensor(img_patch_rgb)
        for c in range(3):
            img_tensor[c] = (img_tensor[c] - DEFAULT_MEAN[c]) / DEFAULT_STD[c]

        # ── 2D keypoints in crop space ──
        kp2d_crop = np.zeros((21, 2), dtype=np.float32)
        for j in range(21):
            kp2d_crop[j] = trans_point2d(joints_2d[j], trans)
        # Normalize to [-0.5, 0.5]
        kp2d_norm = kp2d_crop / self.img_size - 0.5
        kp2d = np.concatenate([kp2d_norm, np.ones((21, 1), dtype=np.float32)], axis=1)  # (21, 3)

        # ── 3D keypoints (camera frame, meters) ──
        joints_3d = gt["joints_cam"].copy()  # (21, 3) OpenPose order
        kp3d = np.concatenate([joints_3d, np.ones((21, 1), dtype=np.float32)], axis=1)  # (21, 4)

        # ── MANO parameters ──
        item = {
            "img": img_tensor,
            "keypoints_2d": kp2d,
            "keypoints_3d": kp3d,
            "mano_params": {
                "global_orient": gt["global_orient_cam"].copy(),
                "hand_pose": gt["hand_pose"].copy(),
                "betas": gt["betas"].copy(),
            },
            "has_mano_params": {
                "global_orient": np.float32(1.0),
                "hand_pose": np.float32(1.0),
                "betas": np.float32(1.0),
            },
            "mano_params_is_axis_angle": {
                "global_orient": np.bool_(True),
                "hand_pose": np.bool_(True),
                "betas": np.bool_(False),
            },
        }

        return item
