"""Hand-pose evaluation metrics.

Naming convention (canonical across this codebase):
  mpjpe_mm       , root-aligned MPJPE  (subtract wrist before computing)
  mpjpe_abs_mm   , absolute-frame MPJPE (no alignment; meaningful only for absolute-output models)
  pa_mpjpe_mm    , Procrustes-aligned MPJPE (frame-invariant)
  pve_mm         , root-aligned per-vertex error
  pa_pve_mm      , Procrustes-aligned per-vertex error
  p2d_px         , 2D reprojection error in pixels
  root_err_mm    , wrist-only absolute offset (degenerate for root-aligned models)
  mrrpe_mm       , Right→Left wrist offset error (both-hand frames only)

Suffix `_v2` means computed on OP_VALID 20-joint subset (used vs Aria MPS GT, since OP[1] is dummy).
"""
import numpy as np


def procrustes_align(P: np.ndarray, T: np.ndarray):
    mu_p, mu_t = P.mean(0), T.mean(0)
    Pc, Tc = P - mu_p, T - mu_t
    U, S, Vt = np.linalg.svd(Pc.T @ Tc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    s = (S * np.diag(D)).sum() / (np.linalg.norm(Pc) ** 2 + 1e-12)
    t = mu_t - s * R @ mu_p
    return (s * (R @ P.T)).T + t, R, float(s), t


def _per_joint_dist_mm(P: np.ndarray, T: np.ndarray) -> float:
    return float(np.linalg.norm(P - T, axis=-1).mean()) * 1000.0


def _root_relative(J: np.ndarray, root: int = 0) -> np.ndarray:
    return J - J[root:root + 1]


def metrics_3d(pred: np.ndarray, gt: np.ndarray, joint_subset: list | None = None) -> dict:
    """All 3D metrics for one (pred, gt) pair, both shape (N, 3) in METERS.

    If joint_subset given, metrics are computed on that subset (e.g. OP_VALID for Aria GT).
    """
    if joint_subset is not None:
        pred_s = pred[joint_subset]
        gt_s = gt[joint_subset]
    else:
        pred_s, gt_s = pred, gt
    return {
        "mpjpe_mm": _per_joint_dist_mm(_root_relative(pred_s), _root_relative(gt_s)),
        "mpjpe_abs_mm": _per_joint_dist_mm(pred_s, gt_s),
        "pa_mpjpe_mm": _per_joint_dist_mm(procrustes_align(pred_s, gt_s)[0], gt_s),
        "root_err_mm": float(np.linalg.norm(pred[0] - gt[0])) * 1000.0,
    }


def metrics_pve(pred_verts: np.ndarray, gt_verts: np.ndarray) -> dict:
    """778-vertex MANO mesh metrics. Inputs in METERS, shape (778, 3)."""
    pr_rel = _root_relative(pred_verts, root=0)  # vertex-0 isn't wrist; just take any consistent reference
    gt_rel = _root_relative(gt_verts, root=0)
    return {
        "pve_mm": _per_joint_dist_mm(pr_rel, gt_rel),
        "pa_pve_mm": _per_joint_dist_mm(procrustes_align(pred_verts, gt_verts)[0], gt_verts),
    }


def metrics_2d(pred_2d: np.ndarray, gt_2d: np.ndarray, joint_subset: list | None = None) -> dict:
    if joint_subset is not None:
        pred_2d, gt_2d = pred_2d[joint_subset], gt_2d[joint_subset]
    return {"p2d_px": float(np.linalg.norm(pred_2d - gt_2d, axis=-1).mean())}


def mrrpe_mm(pred_R: np.ndarray, pred_L: np.ndarray,
             gt_R: np.ndarray, gt_L: np.ndarray) -> float:
    """Mean Relative-Root Position Error. Right→Left wrist offset in absolute camera frame.

    Inputs: each is one wrist 3D position (3,) in METERS.
    """
    pred_off = pred_L - pred_R
    gt_off = gt_L - gt_R
    return float(np.linalg.norm(pred_off - gt_off)) * 1000.0


def aggregate(per_sample: list[dict], group_key: str | None = None) -> dict:
    out = {}
    for k in per_sample[0].keys():
        vals = [s[k] for s in per_sample if k in s and s[k] is not None]
        if not vals or not isinstance(vals[0], (int, float, np.floating, np.integer)):
            continue
        out[k] = float(np.mean(vals))
    out["n"] = len(per_sample)
    if group_key:
        groups = {}
        for s in per_sample:
            groups.setdefault(s.get(group_key), []).append(s)
        out["per_group"] = {g: aggregate(samples) for g, samples in groups.items()}
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    P = rng.standard_normal((21, 3)) * 0.1
    T = P + 0.005 * rng.standard_normal((21, 3))
    print("3D:", metrics_3d(P, T))
    print("PVE:", metrics_pve(rng.standard_normal((778, 3)) * 0.1, rng.standard_normal((778, 3)) * 0.1))
    print("MRRPE:", mrrpe_mm(np.zeros(3), np.array([0.1, 0, 0]), np.zeros(3), np.array([0.11, 0, 0])))
