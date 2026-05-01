"""Joint-order conversions: Aria MPS native, OpenPose, MANO.

The three skeletons used in this study order their 21 keypoints differently.
This module defines the canonical permutations we use to compare predictions
across the four models and against each Aria GT source.
"""
import numpy as np


# OpenPose 21-keypoint hand convention
# 0: wrist
# 1: thumb_cmc, 2: thumb_mcp, 3: thumb_ip, 4: thumb_tip
# 5: index_mcp, ..., 8: index_tip
# 9: middle_mcp, ..., 12: middle_tip
# 13: ring_mcp,  ..., 16: ring_tip
# 17: pinky_mcp, ..., 20: pinky_tip


# Aria MPS native landmark convention (per projectaria_tools):
#   0-4: fingertips (thumb, index, middle, ring, pinky)
#   5: wrist
#   6-7: thumb intermediate + distal
#   8-10: index proximal/intermediate/distal
#   11-13: middle
#   14-16: ring
#   17-19: pinky
#   20: palm center (NO OpenPose analog → skipped)
ARIA_MPS_TO_OPENPOSE = {
    5: 0,    # WRIST
    6: 2,    # THUMB_INTERMEDIATE → THUMB_MCP
    7: 3,    # THUMB_DISTAL → THUMB_IP
    0: 4,    # THUMB_FINGERTIP → THUMB_TIP
    8: 5,    # INDEX_PROXIMAL → INDEX_MCP
    9: 6,
    10: 7,
    1: 8,    # INDEX_FINGERTIP → INDEX_TIP
    11: 9,   # MIDDLE_PROXIMAL
    12: 10,
    13: 11,
    2: 12,
    14: 13,  # RING_PROXIMAL
    15: 14,
    16: 15,
    3: 16,
    17: 17,  # PINKY_PROXIMAL
    18: 18,
    19: 19,
    4: 20,
}
# Aria MPS does NOT have THUMB_CMC → OP[1] is not filled.
# Use OP_VALID = [0, 2..20] (20 indices) for fair metrics.
OP_VALID = sorted(ARIA_MPS_TO_OPENPOSE.values())  # [0, 2, 3, ..., 20], length 20

# MANO (HaMeR/WiLoR/SMPLX joint regressor) → OpenPose 21-keypoint reorder
# Use: joints_op = joints_mano[MANO_TO_OPENPOSE]
MANO_TO_OPENPOSE = np.array(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.int64
)


def aria_mps_to_op(joints_mps: np.ndarray) -> np.ndarray:
    """Remap Aria MPS native (21 entries; idx 20 = palm center) → OpenPose (21 entries; idx 1 = unfilled).

    Output: shape (21, *). Slots that have no Aria source (just OP[1]) are zero-filled.
    """
    out = np.zeros_like(joints_mps)
    for mps_i, op_i in ARIA_MPS_TO_OPENPOSE.items():
        out[op_i] = joints_mps[mps_i]
    return out


def mano_to_op(joints_mano: np.ndarray) -> np.ndarray:
    """Remap MANO joint order → OpenPose. Input/output: (21, *)."""
    return joints_mano[MANO_TO_OPENPOSE]


if __name__ == "__main__":
    print("OP_VALID:", OP_VALID, " len:", len(OP_VALID))
    print("MANO_TO_OPENPOSE:", MANO_TO_OPENPOSE.tolist())
    test = np.arange(21).reshape(21, 1).astype(float).repeat(3, 1)
    print("aria_mps_to_op test (input idx → output idx):")
    out = aria_mps_to_op(test)
    for op_i in range(21):
        src = int(out[op_i, 0]) if out[op_i, 0] != 0 or op_i == 0 else ","
        print(f"  OP[{op_i}] = MPS[{src}]")
