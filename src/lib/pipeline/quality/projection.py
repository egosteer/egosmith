"""Image-projection of world-space hand points and in-frame / off-screen classification."""

from __future__ import annotations

import numpy as np

from .kinematics import transform_points_world_to_camera


def project_points_world_to_image(points_world, extrinsic, intrinsic) -> tuple[np.ndarray, np.ndarray]:
    points_cam = transform_points_world_to_camera(points_world, extrinsic)
    intr = np.asarray(intrinsic, dtype=np.float32).reshape(4)
    uv = np.full((points_cam.shape[0], 2), np.nan, dtype=np.float32)
    valid = np.isfinite(points_cam).all(axis=1) & np.isfinite(intr).all() & (points_cam[:, 2] > 1e-6)
    if np.any(valid):
        uv[valid, 0] = intr[0] * points_cam[valid, 0] / points_cam[valid, 2] + intr[2]
        uv[valid, 1] = intr[1] * points_cam[valid, 1] / points_cam[valid, 2] + intr[3]
    valid &= np.isfinite(uv).all(axis=1)
    return uv, valid


def classify_hand_projection(points_world, extrinsic, intrinsic, image_size, *, severe_offscreen_scale: float) -> dict:
    width, height = int(image_size[0]), int(image_size[1])
    uv, valid = project_points_world_to_image(points_world, extrinsic, intrinsic)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image_size: {image_size}")

    inframe = (
        valid
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(height))
    )

    margin_x = max(0.0, float(severe_offscreen_scale) - 1.0) * float(width)
    margin_y = max(0.0, float(severe_offscreen_scale) - 1.0) * float(height)
    severe_bounds = (
        valid
        & (uv[:, 0] >= -margin_x)
        & (uv[:, 0] < float(width) + margin_x)
        & (uv[:, 1] >= -margin_y)
        & (uv[:, 1] < float(height) + margin_y)
    )

    return {
        "uv": uv,
        "valid": valid,
        "any_point_inframe": bool(np.any(inframe)),
        "all_points_out_of_frame": bool(np.all(~inframe)),
        "all_points_severe_offscreen": bool(np.all(~severe_bounds)),
    }
