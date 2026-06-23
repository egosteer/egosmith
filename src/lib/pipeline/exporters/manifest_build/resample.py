"""Resampling helpers for manifest build/export."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from lib.pipeline.exporters.webdataset_features import InvalidCameraDataError
from lib.pipeline.exporters.webdataset_geometry import axis_angle_to_rot6d


def build_source_target_times(source_count: int, target_count: int, source_fps: float, target_fps: float):
    if source_count <= 0 or target_count <= 0:
        raise ValueError(f"Invalid counts for resampling: source={source_count}, target={target_count}")
    if source_count == 1:
        return np.zeros((1,), dtype=np.float64), np.zeros((target_count,), dtype=np.float64)

    if source_fps > 0 and target_fps > 0:
        source_times = np.arange(source_count, dtype=np.float64) / float(source_fps)
        target_times = np.arange(target_count, dtype=np.float64) / float(target_fps)
    else:
        source_times = np.arange(source_count, dtype=np.float64)
        target_times = np.linspace(0.0, float(source_count - 1), num=target_count, dtype=np.float64)
    return source_times, np.clip(target_times, source_times[0], source_times[-1])


def resample_linear_sequence(sequence, target_count: int, source_fps: float, target_fps: float) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float32)
    source_count = int(array.shape[0])
    if target_count == source_count:
        return array.astype(np.float32, copy=False)
    if source_count == 1:
        return np.repeat(array[:1], target_count, axis=0).astype(np.float32, copy=False)

    source_times, target_times = build_source_target_times(source_count, target_count, source_fps, target_fps)
    flat = array.reshape(source_count, -1)
    output = np.empty((target_count, flat.shape[1]), dtype=np.float32)
    for column_idx in range(flat.shape[1]):
        output[:, column_idx] = np.interp(target_times, source_times, flat[:, column_idx]).astype(np.float32)
    return output.reshape((target_count,) + array.shape[1:])


def resample_nearest_sequence(sequence, target_count: int, source_fps: float, target_fps: float) -> np.ndarray:
    array = np.asarray(sequence)
    source_count = int(array.shape[0])
    if target_count == source_count:
        return array
    if source_count == 1:
        return np.repeat(array[:1], target_count, axis=0)

    source_times, target_times = build_source_target_times(source_count, target_count, source_fps, target_fps)
    float_indices = np.interp(target_times, source_times, np.arange(source_count, dtype=np.float64))
    nearest_indices = np.clip(np.rint(float_indices).astype(np.int64), 0, source_count - 1)
    return array[nearest_indices]


def resample_axis_angle_batch(axis_angle, target_count: int, source_fps: float, target_fps: float) -> np.ndarray:
    array = np.asarray(axis_angle, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected axis-angle batch with shape (N,T,3), got {array.shape}")
    batch_size, source_count, _ = array.shape
    if target_count == source_count:
        return array.astype(np.float32, copy=False)
    if source_count == 1:
        return np.repeat(array[:, :1, :], target_count, axis=1).astype(np.float32, copy=False)

    source_times, target_times = build_source_target_times(source_count, target_count, source_fps, target_fps)
    output = np.empty((batch_size, target_count, 3), dtype=np.float32)
    for batch_idx in range(batch_size):
        rotations = Rotation.from_rotvec(array[batch_idx])
        slerp = Slerp(source_times, rotations)
        output[batch_idx] = slerp(target_times).as_rotvec().astype(np.float32)
    return output


def resample_extrinsics_sequence(extrinsics, target_count: int, source_fps: float, target_fps: float) -> np.ndarray:
    mats = np.asarray(extrinsics, dtype=np.float32)
    if not np.isfinite(mats).all():
        raise InvalidCameraDataError("extrinsics contain non-finite values before resampling")
    source_count = int(mats.shape[0])
    if target_count == source_count:
        return mats.astype(np.float32, copy=False)
    if source_count == 1:
        return np.repeat(mats[:1], target_count, axis=0).astype(np.float32, copy=False)

    source_times, target_times = build_source_target_times(source_count, target_count, source_fps, target_fps)
    if not np.isfinite(mats[:, :3, :3]).all():
        raise InvalidCameraDataError("rotation blocks contain non-finite values before resampling")
    try:
        rotations = Rotation.from_matrix(mats[:, :3, :3])
        slerp = Slerp(source_times, rotations)
        interp_rot = slerp(target_times).as_matrix().astype(np.float32)
    except Exception as error:
        raise InvalidCameraDataError(f"failed to resample camera rotations: {error}") from error
    interp_trans = resample_linear_sequence(mats[:, :3, 3], target_count, source_fps, target_fps)

    output = np.tile(np.eye(4, dtype=np.float32), (target_count, 1, 1))
    output[:, :3, :3] = interp_rot
    output[:, :3, 3] = interp_trans
    return output


def resample_episode_features(
    wrist_state,
    hand_state,
    pred_rot,
    pred_hand_pose,
    pred_betas,
    extrinsics,
    presence_per_frame,
    target_count: int,
    *,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
):
    source_count = int(wrist_state.shape[0])
    if not interpolate_labels:
        frame_count = min(source_count, target_count)
        return (
            wrist_state[:frame_count],
            hand_state[:frame_count],
            pred_rot[:, :frame_count],
            pred_hand_pose[:, :frame_count],
            pred_betas[:, :frame_count],
            np.asarray(extrinsics[:frame_count], dtype=np.float32),
            np.asarray(presence_per_frame[:frame_count]),
        )

    if target_count <= 0:
        raise ValueError(f"Invalid target_count for resampling: {target_count}")

    wrist_positions = resample_linear_sequence(wrist_state[:, :6].cpu().numpy(), target_count, source_fps, target_fps)
    hand_state_resampled = resample_linear_sequence(hand_state.cpu().numpy(), target_count, source_fps, target_fps)
    pred_rot_resampled = resample_axis_angle_batch(pred_rot.float().cpu().numpy(), target_count, source_fps, target_fps)
    pred_hand_pose_resampled = resample_axis_angle_batch(
        pred_hand_pose.float().cpu().numpy().reshape(-1, source_count, 3),
        target_count,
        source_fps,
        target_fps,
    ).reshape(2, target_count, 45)
    pred_betas_resampled = resample_linear_sequence(
        pred_betas.float().cpu().numpy().transpose(1, 0, 2),
        target_count,
        source_fps,
        target_fps,
    ).transpose(1, 0, 2)
    rot6d = axis_angle_to_rot6d(torch.from_numpy(pred_rot_resampled)).cpu().numpy().astype(np.float32)
    wrist_state_resampled = np.concatenate(
        [
            wrist_positions,
            rot6d[0],
            rot6d[1],
        ],
        axis=-1,
    ).astype(np.float32)
    extrinsics_resampled = resample_extrinsics_sequence(extrinsics, target_count, source_fps, target_fps)
    presence_resampled = resample_nearest_sequence(presence_per_frame, target_count, source_fps, target_fps)
    return (
        torch.from_numpy(wrist_state_resampled),
        torch.from_numpy(hand_state_resampled.astype(np.float32)),
        torch.from_numpy(pred_rot_resampled.astype(np.float32)),
        torch.from_numpy(pred_hand_pose_resampled.astype(np.float32)),
        torch.from_numpy(pred_betas_resampled.astype(np.float32)),
        extrinsics_resampled.astype(np.float32, copy=False),
        np.asarray(presence_resampled),
    )
