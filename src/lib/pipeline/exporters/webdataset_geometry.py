"""Geometry helpers for WebDataset export."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def axis_angle_to_rot6d(axis_angle):
    """Convert axis-angle (*, 3) to 6D rotation representation (*, 6)."""
    from hawor.utils.geometry import aa_to_rotmat

    orig_shape = axis_angle.shape[:-1]
    flat = axis_angle.reshape(-1, 3)
    rotmat = aa_to_rotmat(flat)
    rot6d = rotmat[:, :, :2].permute(0, 2, 1).contiguous().reshape(-1, 6)
    return rot6d.reshape(*orig_shape, 6)


def quat_to_4x4(traj_row, scale):
    """Convert SLAM traj row [tx, ty, tz, qx, qy, qz, qw] to 4x4 c2w matrix."""
    t = traj_row[:3] * scale
    quat_xyzw = traj_row[3:7]
    rot = Rotation.from_quat(quat_xyzw).as_matrix()
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot
    mat[:3, 3] = t
    return mat


def interpolate_extrinsics(tstamps, traj, scale, total_frames):
    """Interpolate SLAM keyframe extrinsics to all frames as world-to-camera matrices."""
    tstamps = tstamps.astype(np.float64)
    num_keyframes = len(tstamps)

    if num_keyframes == 0:
        return np.tile(np.eye(4, dtype=np.float32), (total_frames, 1, 1))

    if num_keyframes == 1:
        mat = quat_to_4x4(traj[0], scale)
        return np.tile(np.linalg.inv(mat).astype(np.float32), (total_frames, 1, 1))

    translations = traj[:, :3] * scale
    quats_xyzw = traj[:, 3:7]
    rotations = Rotation.from_quat(quats_xyzw)

    slerp = Slerp(tstamps, rotations)
    all_frames = np.arange(total_frames, dtype=np.float64)
    all_frames_clamped = np.clip(all_frames, tstamps[0], tstamps[-1])

    interp_rots = slerp(all_frames_clamped).as_matrix()
    interp_trans = np.stack(
        [np.interp(all_frames_clamped, tstamps, translations[:, i]) for i in range(3)],
        axis=-1,
    )

    mats = np.zeros((total_frames, 4, 4), dtype=np.float32)
    mats[:, :3, :3] = interp_rots
    mats[:, :3, 3] = interp_trans
    mats[:, 3, 3] = 1.0
    return np.linalg.inv(mats).astype(np.float32)


def normalize_slam_keyframes(tstamps, traj):
    """Align SLAM keyframe timestamps with trajectory rows, tolerating dirty data."""
    tstamps = np.asarray(tstamps, dtype=np.int64).reshape(-1)
    traj = np.asarray(traj)

    if len(tstamps) == 0 or len(traj) == 0:
        return tstamps[:0], traj[:0]

    if len(traj) == len(tstamps):
        aligned_tstamps = tstamps
        aligned_traj = traj
    else:
        zero_based_valid = (tstamps >= 0) & (tstamps < len(traj))
        one_based = tstamps - 1
        one_based_valid = (one_based >= 0) & (one_based < len(traj))

        if not zero_based_valid.all() and one_based_valid.all():
            aligned_tstamps = tstamps
            aligned_traj = traj[one_based]
        else:
            aligned_tstamps = tstamps[zero_based_valid]
            aligned_traj = traj[aligned_tstamps]

    if len(aligned_tstamps) == 0:
        return aligned_tstamps, aligned_traj

    order = np.argsort(aligned_tstamps, kind="stable")
    aligned_tstamps = aligned_tstamps[order]
    aligned_traj = aligned_traj[order]

    keep = np.ones(len(aligned_tstamps), dtype=bool)
    keep[1:] = aligned_tstamps[1:] != aligned_tstamps[:-1]
    return aligned_tstamps[keep], aligned_traj[keep]
