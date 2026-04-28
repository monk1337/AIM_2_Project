"""Debug WiLoR detection on Aria val, see what comes back from YOLO."""
import sys
import numpy as np
sys.path.insert(0, "/workspace/code")
from eval_aria_loader import load_aria_val, derive_bbox_from_joints2d

import torch
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

pipe = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float32, verbose=False)

rows = load_aria_val()[:10]
for r in rows[:5]:
    img = np.asarray(r["image"])
    print(f"\n=== {r['sequence_name']}/{r['frame_id']} {r['hand_side']} ===")
    print(f"  image: shape={img.shape} dtype={img.dtype} mean={img.mean():.1f}")
    print(f"  GT joints_2d range: x=[{r['joints_2d_hamersam'][:,0].min():.0f},{r['joints_2d_hamersam'][:,0].max():.0f}] "
          f"y=[{r['joints_2d_hamersam'][:,1].min():.0f},{r['joints_2d_hamersam'][:,1].max():.0f}]")
    print(f"  GT bbox: {derive_bbox_from_joints2d(r['joints_2d_hamersam'])}")
    # Try various confidence thresholds
    for conf in [0.05, 0.1, 0.3]:
        det = pipe.predict(img, hand_conf=conf, rescale_factor=2.5)
        print(f"  conf={conf}: {len(det)} detections", [(d['hand_bbox'], 'right' if d['is_right'] else 'left') for d in det])
