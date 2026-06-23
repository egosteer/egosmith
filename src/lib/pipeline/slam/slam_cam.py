"""SLAM camera-trajectory loading and dense-export validation (EgoSmith first-party).

These helpers read EgoSmith's DPVO SLAM export (``hawor_slam_w_scale`` npz: scaled
``traj`` + ``scale`` + optional ``tstamp``) and turn it into per-frame camera poses
for the infiller and visualization. They diverge from HaWoR's originals: dtype is
pinned to float32 (avoids a float64-vs-float32 einsum error when ``scale`` promotes),
and interpolation is replaced by a hard requirement that the export already be dense
per video frame (sparse keyframe exports are rejected, not silently interpolated).

The verbatim HaWoR quaternion->matrix helper is imported from the obtained
``lib/eval_utils/custom_utils.py`` rather than duplicated here.
"""

import numpy as np
import torch

from lib.eval_utils.custom_utils import quaternion_to_matrix


def load_slam_cam(fpath):
    print(f"Loading cameras from {fpath}...")
    pred_cam = dict(np.load(fpath, allow_pickle=True))
    pred_traj = pred_cam["traj"]
    # Use float32 everywhere: traj may be float32 in npz while np.float64(scale) promotes
    # (tensor * scale) to float64, but quaternion_to_matrix stays float32 → einsum dtype error.
    scale = float(pred_cam["scale"])
    t_c2w_sla = torch.as_tensor(pred_traj[:, :3], dtype=torch.float32) * scale
    pred_camq = torch.as_tensor(pred_traj[:, 3:], dtype=torch.float32)
    R_c2w_sla = quaternion_to_matrix(pred_camq[:, [3, 0, 1, 2]])
    R_w2c_sla = R_c2w_sla.transpose(-1, -2)
    t_w2c_sla = -torch.einsum("bij,bj->bi", R_w2c_sla, t_c2w_sla)
    return R_w2c_sla, t_w2c_sla, R_c2w_sla, t_c2w_sla


def validate_dense_slam_export(fpath):
    """
    Validate that hawor_slam_w_scale export is already dense per video frame.
    Old sparse DPVO exports must be repaired before infiller runs.
    """
    pred_cam = dict(np.load(fpath, allow_pickle=False))
    pred_traj = pred_cam["traj"]
    tstamp = np.asarray(pred_cam.get("tstamp", np.arange(len(pred_traj)))).astype(
        np.int64
    ).reshape(-1)

    dense_by_video_frame = pred_traj.shape[0] != tstamp.shape[0]
    dense_by_contiguous_tstamp = pred_traj.shape[0] == tstamp.shape[0] and np.array_equal(
        tstamp,
        np.arange(pred_traj.shape[0], dtype=np.int64),
    )
    if not dense_by_video_frame and not dense_by_contiguous_tstamp:
        raise RuntimeError(
            f"DPVO infiller requires dense per-frame SLAM cameras; interpolation is disabled for {fpath}. "
            "Repair old exports first, e.g. with "
            "deprecated/unrefactored_tools/scripts/repair_dpvo_dense_slam_exports.py."
        )


def interpolate_slam_cameras_at_video_frames(fpath, video_frame_indices):
    """
    Get c2w R, t by video frame index.
    Only per-frame dense trajectories are supported; if keyframe interpolation is needed, raise an error so the caller marks the video as failed.
    """
    pred_cam = dict(np.load(fpath, allow_pickle=True))
    pred_traj = pred_cam["traj"]
    scale = float(pred_cam["scale"])
    tstamp = np.asarray(pred_cam.get("tstamp", np.arange(len(pred_traj)))).astype(
        np.int64
    ).reshape(-1)
    t_c2w = (pred_traj[:, :3] * scale).astype(np.float64)
    pred_camq = torch.tensor(pred_traj[:, 3:])
    R_c2w = quaternion_to_matrix(pred_camq[:, [3, 0, 1, 2]]).numpy()

    vf = np.asarray(video_frame_indices, dtype=np.int64).reshape(-1)

    validate_dense_slam_export(fpath)

    if pred_traj.shape[0] <= 0:
        raise ValueError("empty SLAM trajectory")
    vf_clipped = np.clip(vf, 0, pred_traj.shape[0] - 1)
    return (
        torch.tensor(R_c2w[vf_clipped], dtype=torch.float32),
        torch.tensor(t_c2w[vf_clipped], dtype=torch.float32),
    )
