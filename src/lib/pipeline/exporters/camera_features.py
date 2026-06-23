"""Per-episode camera features: load SLAM trajectory -> per-frame extrinsics + intrinsic.

Self-contained: rate-limited SLAM warnings, finite-value guards, and the InvalidCameraDataError
raised when camera data is present but too dirty to trust.
"""

import os
from pathlib import Path

import numpy as np

from .webdataset_geometry import (
    interpolate_extrinsics,
    normalize_slam_keyframes,
    quat_to_4x4,
)

DEFAULT_INTRINSIC = np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float32)
_SLAM_WARNING_COUNTS = {}


def _log_slam_warning(kind: str, message: str):
    count = int(_SLAM_WARNING_COUNTS.get(kind, 0)) + 1
    _SLAM_WARNING_COUNTS[kind] = count
    if count <= 10 or count in (20, 50, 100) or count % 500 == 0:
        suffix = "" if count == 1 else f" [count={count}]"
        print(f"  Warning: {message}{suffix}")


class InvalidCameraDataError(ValueError):
    """Raised when episode camera data is present but too dirty to trust."""


def _ensure_finite_array(name: str, value):
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise InvalidCameraDataError(f"{name} contains non-finite values")
    return array


def _load_episode_camera_features(ep, num_frames):
    slam_dir = os.path.join(ep["crop_dir"], "SLAM")
    extrinsics = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
    intrinsic = DEFAULT_INTRINSIC.copy()

    slam_files = sorted(Path(slam_dir).glob("hawor_slam_w_scale_*.npz")) if os.path.isdir(slam_dir) else []
    if not slam_files:
        return extrinsics, intrinsic

    try:
        slam_data = np.load(str(slam_files[0]), allow_pickle=True)
        tstamps = np.asarray(slam_data.get("tstamp", np.arange(len(slam_data["traj"]))), dtype=np.int64).reshape(-1)
        traj = _ensure_finite_array("traj", np.asarray(slam_data["traj"], dtype=np.float32))
        scale = float(slam_data["scale"])
        img_focal = float(slam_data["img_focal"])
        img_center = _ensure_finite_array("img_center", np.asarray(slam_data["img_center"], dtype=np.float32))
        if not np.isfinite(scale):
            raise InvalidCameraDataError("scale is non-finite")
        if not np.isfinite(img_focal):
            raise InvalidCameraDataError("img_focal is non-finite")

        intrinsic = np.array(
            [
                img_focal,
                img_focal,
                float(img_center[0]),
                float(img_center[1]),
            ],
            dtype=np.float32,
        )
        _ensure_finite_array("intrinsic", intrinsic)

        if int(traj.shape[0]) == int(num_frames):
            c2w = np.stack([quat_to_4x4(traj_row, scale) for traj_row in traj], axis=0)
            _ensure_finite_array("c2w", c2w)
            extrinsics = np.linalg.inv(c2w).astype(np.float32)
            _ensure_finite_array("direct_extrinsics", extrinsics)
        else:
            tstamps, traj = normalize_slam_keyframes(tstamps, traj)
            if len(tstamps) <= 0:
                raise InvalidCameraDataError("empty normalized SLAM trajectory")
            extrinsics = interpolate_extrinsics(tstamps, traj, scale, int(num_frames))
            _ensure_finite_array("interpolated_extrinsics", extrinsics)
            _log_slam_warning(
                "frame_count_mismatch_interpolate",
                f"SLAM/frame count mismatch for {ep['episode_id']}: "
                f"traj={traj.shape[0]} num_frames={num_frames}; "
                "used timestamp interpolation instead of direct traj repeat/truncate.",
            )
    except InvalidCameraDataError:
        raise
    except Exception as error:
        raise InvalidCameraDataError(str(error)) from error

    return extrinsics, intrinsic
