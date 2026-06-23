"""Helpers for native depth stage artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


NATIVE_DEPTH_STAGE_NAME = "native_depth"
NATIVE_DEPTH_DIRNAME = "NATIVE_DEPTH"
NATIVE_DEPTH_FILENAME = "any4d_depth.npz"


def get_native_depth_output_path(seq_folder: str | Path) -> Path:
    return Path(seq_folder) / NATIVE_DEPTH_DIRNAME / NATIVE_DEPTH_FILENAME


def validate_native_depth_output(
    seq_folder: str | Path,
    *,
    expected_frame_count: int | None = None,
) -> dict:
    path = get_native_depth_output_path(seq_folder)
    if not path.is_file():
        raise FileNotFoundError(f"native depth artifact not found: {path}")

    with np.load(path, allow_pickle=False) as payload:
        required_keys = {"depths", "frame_indices", "intrinsics", "camera_poses"}
        missing = sorted(required_keys - set(payload.files))
        if missing:
            raise ValueError(f"native depth artifact missing keys: {missing}")

        depths = np.asarray(payload["depths"])
        frame_indices = np.asarray(payload["frame_indices"])
        intrinsics = np.asarray(payload["intrinsics"])
        camera_poses = np.asarray(payload["camera_poses"])

    if depths.ndim != 3:
        raise ValueError(f"native depth depths must have shape (T, H, W), got {depths.shape}")
    if frame_indices.ndim != 1:
        raise ValueError(f"native depth frame_indices must have shape (T,), got {frame_indices.shape}")
    if intrinsics.shape != (depths.shape[0], 3, 3):
        raise ValueError(
            f"native depth intrinsics must have shape {(depths.shape[0], 3, 3)}, got {intrinsics.shape}"
        )
    if camera_poses.shape != (depths.shape[0], 4, 4):
        raise ValueError(
            f"native depth camera_poses must have shape {(depths.shape[0], 4, 4)}, got {camera_poses.shape}"
        )
    if frame_indices.shape[0] != depths.shape[0]:
        raise ValueError(
            f"native depth frame_indices length {frame_indices.shape[0]} != depth count {depths.shape[0]}"
        )
    if expected_frame_count is not None and depths.shape[0] != int(expected_frame_count):
        raise ValueError(
            f"native depth frame count {depths.shape[0]} != expected {int(expected_frame_count)}"
        )
    if not np.isfinite(depths).all():
        raise ValueError("native depth contains non-finite values")
    if not np.isfinite(intrinsics).all():
        raise ValueError("native depth intrinsics contain non-finite values")
    if not np.isfinite(camera_poses).all():
        raise ValueError("native depth camera_poses contain non-finite values")
    if np.any(intrinsics[:, 0, 0] <= 0.0) or np.any(intrinsics[:, 1, 1] <= 0.0):
        raise ValueError("native depth intrinsics contain non-positive focal lengths")
    if np.any(depths < 0.0):
        raise ValueError("native depth contains negative depth values")

    return {
        "path": str(path),
        "frame_count": int(depths.shape[0]),
        "height": int(depths.shape[1]),
        "width": int(depths.shape[2]),
        "dtype": str(depths.dtype),
    }
