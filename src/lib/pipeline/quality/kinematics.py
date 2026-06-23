"""Geometry and per-frame motion-step computations over lowdim components.

Translation/rotation steps, world->camera transforms, camera-space abs/axis extents, finger-in-wrist
frame, and the sliding-window wrist-relative-to-camera extremes (paper Stage-4 chunk level).
"""

from __future__ import annotations

import numpy as np

from .lowdim import _rot6d_to_rotmat


def max_translation_step(sequence, valid_mask=None) -> dict:
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected (T, C) sequence, got shape {array.shape}")
    if array.shape[0] < 2:
        return {
            "max_step": 0.0,
            "pair_index": None,
            "valid_pairs": 0,
        }

    diffs = np.linalg.norm(array[1:] - array[:-1], axis=1)
    pair_mask = np.ones(diffs.shape[0], dtype=bool)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid.shape[0] != array.shape[0]:
            raise ValueError(
                f"valid_mask length mismatch: expected {array.shape[0]}, got {valid.shape[0]}"
            )
        pair_mask &= valid[1:] & valid[:-1]

    valid_indices = np.flatnonzero(pair_mask)
    if valid_indices.size == 0:
        return {
            "max_step": 0.0,
            "pair_index": None,
            "valid_pairs": 0,
        }

    candidate_diffs = diffs[valid_indices]
    local_argmax = int(np.argmax(candidate_diffs))
    pair_index = int(valid_indices[local_argmax])
    return {
        "max_step": float(candidate_diffs[local_argmax]),
        "pair_index": pair_index,
        "valid_pairs": int(valid_indices.size),
    }


def max_camera_step(extrinsics, valid_mask=None) -> dict:
    mats = np.asarray(extrinsics, dtype=np.float32)
    if mats.ndim == 2 and mats.shape[1] == 16:
        mats = mats.reshape(-1, 4, 4)
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError(f"Expected (T,4,4) extrinsics, got shape {mats.shape}")
    if mats.shape[0] < 2:
        return {
            "max_translation_step": 0.0,
            "translation_pair_index": None,
            "max_rotation_step": 0.0,
            "rotation_pair_index": None,
            "valid_pairs": 0,
        }

    translations = mats[:, :3, 3]
    rotations = mats[:, :3, :3]
    translation_diffs = np.linalg.norm(translations[1:] - translations[:-1], axis=1)
    rotation_diffs = np.linalg.norm((rotations[1:] - rotations[:-1]).reshape(rotations.shape[0] - 1, -1), axis=1)

    pair_mask = np.ones(translation_diffs.shape[0], dtype=bool)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid.shape[0] != mats.shape[0]:
            raise ValueError(
                f"valid_mask length mismatch: expected {mats.shape[0]}, got {valid.shape[0]}"
            )
        pair_mask &= valid[1:] & valid[:-1]

    valid_indices = np.flatnonzero(pair_mask)
    if valid_indices.size == 0:
        return {
            "max_translation_step": 0.0,
            "translation_pair_index": None,
            "max_rotation_step": 0.0,
            "rotation_pair_index": None,
            "valid_pairs": 0,
        }

    translation_local_argmax = int(np.argmax(translation_diffs[valid_indices]))
    rotation_local_argmax = int(np.argmax(rotation_diffs[valid_indices]))
    translation_pair_index = int(valid_indices[translation_local_argmax])
    rotation_pair_index = int(valid_indices[rotation_local_argmax])
    return {
        "max_translation_step": float(translation_diffs[translation_pair_index]),
        "translation_pair_index": translation_pair_index,
        "max_rotation_step": float(rotation_diffs[rotation_pair_index]),
        "rotation_pair_index": rotation_pair_index,
        "valid_pairs": int(valid_indices.size),
    }


def transform_points_world_to_camera(points, extrinsic) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    mat = np.asarray(extrinsic, dtype=np.float32).reshape(4, 4)
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    return (pts @ rot.T) + trans


def camera_space_abs_metrics(points_world, extrinsic) -> dict:
    points_cam = transform_points_world_to_camera(points_world, extrinsic)
    abs_points = np.abs(points_cam)
    return {
        "max_abs": float(abs_points.max()) if abs_points.size else 0.0,
        "max_abs_x": float(abs_points[:, 0].max()) if abs_points.size else 0.0,
        "max_abs_y": float(abs_points[:, 1].max()) if abs_points.size else 0.0,
        "max_abs_z": float(abs_points[:, 2].max()) if abs_points.size else 0.0,
    }


def camera_space_axis_metrics(points_world, extrinsic) -> dict:
    points_cam = transform_points_world_to_camera(points_world, extrinsic)
    if points_cam.size == 0:
        return {
            "min_x": 0.0,
            "max_x": 0.0,
            "min_y": 0.0,
            "max_y": 0.0,
            "min_z": 0.0,
            "max_z": 0.0,
        }
    return {
        "min_x": float(points_cam[:, 0].min()),
        "max_x": float(points_cam[:, 0].max()),
        "min_y": float(points_cam[:, 1].min()),
        "max_y": float(points_cam[:, 1].max()),
        "min_z": float(points_cam[:, 2].min()),
        "max_z": float(points_cam[:, 2].max()),
    }


def _abs_axis_metrics(points_local) -> tuple[dict, dict]:
    """abs + per-axis min/max for points already expressed in their target frame (no camera
    transform). Output mirrors camera_space_abs_metrics / camera_space_axis_metrics."""
    pts = np.asarray(points_local, dtype=np.float32).reshape(-1, 3)
    if pts.size == 0:
        zero_axis = {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0, "min_z": 0.0, "max_z": 0.0}
        return {"max_abs": 0.0}, zero_axis
    axis_metrics = {
        "min_x": float(pts[:, 0].min()),
        "max_x": float(pts[:, 0].max()),
        "min_y": float(pts[:, 1].min()),
        "max_y": float(pts[:, 1].max()),
        "min_z": float(pts[:, 2].min()),
        "max_z": float(pts[:, 2].max()),
    }
    return {"max_abs": float(np.abs(pts).max())}, axis_metrics


def _fingers_relative_to_wrist(fingertips, wrist_translation, wrist_rot6d) -> np.ndarray:
    """Finger keypoints in the wrist's own frame: R_wrist^T (finger_world - wrist_world).

    Paper Stage-4 chunk level: "finger joints relative to the wrist". R columns are [b1,b2,b3]
    (see _rot6d_to_rotmat), so (P - t) @ R gives each row's coords in the wrist basis = R^T(p-t).
    """
    rot = _rot6d_to_rotmat(wrist_rot6d)
    rel = np.asarray(fingertips, dtype=np.float32).reshape(-1, 3) - np.asarray(
        wrist_translation, dtype=np.float32
    ).reshape(3)
    return (rel @ rot).astype(np.float32)


def windowed_wrist_camera_extremes(frame_indices, wrist_world, extrinsics, *, past_frames, future_frames):
    """Sliding-window wrist-relative-to-camera extremes (paper Stage-4 chunk level).

    For each buffered frame t, transform every in-window wrist position (window
    [frame_idx[t] - past_frames, frame_idx[t] + future_frames]) into frame t's camera frame and
    accumulate per-axis min/max and max-abs over all (t, in-window) pairs.

    Returns (axis_min[3], axis_max[3], max_abs); empty input -> zeros.
    """
    n = len(frame_indices)
    if n == 0:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 0.0
    idx = np.asarray(frame_indices, dtype=np.int64)
    wrist = np.asarray(wrist_world, dtype=np.float32).reshape(n, -1, 3)
    axis_min = np.full(3, np.inf, dtype=np.float64)
    axis_max = np.full(3, -np.inf, dtype=np.float64)
    max_abs = 0.0
    for t in range(n):
        sel = (idx >= idx[t] - int(past_frames)) & (idx <= idx[t] + int(future_frames))
        window_world = wrist[sel].reshape(-1, 3)
        mat = np.asarray(extrinsics[t], dtype=np.float32).reshape(4, 4)
        cam = window_world @ mat[:3, :3].T + mat[:3, 3]
        axis_min = np.minimum(axis_min, cam.min(axis=0))
        axis_max = np.maximum(axis_max, cam.max(axis=0))
        max_abs = max(max_abs, float(np.abs(cam).max()))
    return axis_min.astype(np.float32), axis_max.astype(np.float32), float(max_abs)
