"""Depth/action consistency summaries for final WebDataset validation."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from lib.pipeline.exporters.webdataset_rewriter import iter_shard_paths, iter_shard_samples
from lib.pipeline.quality.quality_metrics import decode_lowdim


STATE_WRIST_SLICE = slice(0, 6)
STATE_FINGERTIP_SLICE = slice(18, 48)
EXTRINSIC_SLICE = slice(96, 112)
INTRINSIC_SLICE = slice(112, 116)


def _decode_depth(depth_bytes: bytes) -> np.ndarray:
    depth = np.load(io.BytesIO(depth_bytes), allow_pickle=False)
    depth = np.asarray(depth)
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) * 1e-3
    return depth.astype(np.float32)


def _hand_points_from_lowdim(lowdim: np.ndarray) -> np.ndarray:
    wrist = np.asarray(lowdim[STATE_WRIST_SLICE], dtype=np.float32).reshape(2, 3)
    fingertips = np.asarray(lowdim[STATE_FINGERTIP_SLICE], dtype=np.float32).reshape(2, 5, 3).reshape(10, 3)
    return np.concatenate([wrist, fingertips], axis=0)


def _project_points(points_world: np.ndarray, lowdim: np.ndarray, depth_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    w2c = np.asarray(lowdim[EXTRINSIC_SLICE], dtype=np.float32).reshape(4, 4)
    fx, fy, cx, cy = np.asarray(lowdim[INTRINSIC_SLICE], dtype=np.float32).reshape(4)
    points_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=1)
    points_cam = (w2c @ points_h.T).T[:, :3]
    z = points_cam[:, 2]
    valid_z = np.isfinite(z) & (z > 1e-6)
    u = fx * (points_cam[:, 0] / np.maximum(z, 1e-6)) + cx
    v = fy * (points_cam[:, 1] / np.maximum(z, 1e-6)) + cy
    pixels = np.stack([u, v], axis=1)
    height, width = depth_shape
    rounded = np.rint(pixels).astype(np.int64)
    in_bounds = (
        valid_z
        & np.isfinite(pixels).all(axis=1)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    return rounded[in_bounds], z[in_bounds]


def analyze_depth_action_consistency(
    *,
    dataset_dir: str,
    sample_limit: int | None = None,
    max_examples: int = 32,
) -> dict:
    """Compare exported depth against hand keypoints projected from lowdim.

    Iterates WebDataset shards under ``dataset_dir``, projects each sample's
    hand points into the depth image, and measures the absolute error between
    sampled depth and projected z. ``sample_limit`` caps total samples scanned;
    ``max_examples`` caps how many per-sample example/error records are kept.

    Returns a report dict with keys: ``dataset_dir``, ``samples_total``,
    ``samples_with_depth``, ``samples_checked``, ``points_compared``,
    ``missing_depth_samples``, ``projection_empty_samples``,
    ``decode_failures``, ``examples`` (list), and ``abs_error_m`` (a
    mean/median/p90/p95/max summary, empty when nothing was compared).
    """
    shard_paths = list(iter_shard_paths(dataset_dir))
    report = {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "samples_total": 0,
        "samples_with_depth": 0,
        "samples_checked": 0,
        "points_compared": 0,
        "missing_depth_samples": 0,
        "projection_empty_samples": 0,
        "decode_failures": 0,
        "abs_error_m": {},
        "examples": [],
    }
    errors = []

    limit_reached = False
    for shard_path in shard_paths:
        if limit_reached:
            break
        shard_name = Path(shard_path).name
        for sample in iter_shard_samples(shard_path):
            if sample_limit is not None and report["samples_total"] >= int(sample_limit):
                limit_reached = True
                break
            report["samples_total"] += 1
            sample_key = sample["key"]
            if sample.get("depth_bytes") is None:
                report["missing_depth_samples"] += 1
                continue
            report["samples_with_depth"] += 1
            try:
                lowdim = decode_lowdim(sample["lowdim_bytes"])
                depth = _decode_depth(sample["depth_bytes"])
                if depth.ndim != 2 or not np.isfinite(depth).all():
                    raise ValueError(f"invalid depth shape/values: {depth.shape}")
                points = _hand_points_from_lowdim(lowdim)
                pixels, projected_z = _project_points(points, lowdim, depth.shape)
                if pixels.shape[0] == 0:
                    report["projection_empty_samples"] += 1
                    continue
                sampled_depth = depth[pixels[:, 1], pixels[:, 0]]
                valid = np.isfinite(sampled_depth) & (sampled_depth > 0.0) & np.isfinite(projected_z)
                if not valid.any():
                    report["projection_empty_samples"] += 1
                    continue
                abs_error = np.abs(sampled_depth[valid] - projected_z[valid])
                errors.extend(float(value) for value in abs_error.tolist())
                report["points_compared"] += int(abs_error.shape[0])
                report["samples_checked"] += 1
                if len(report["examples"]) < max_examples:
                    report["examples"].append(
                        {
                            "sample_key": sample_key,
                            "shard_name": shard_name,
                            "points": int(abs_error.shape[0]),
                            "mean_abs_error_m": float(np.mean(abs_error)),
                            "max_abs_error_m": float(np.max(abs_error)),
                        }
                    )
            except Exception as error:
                report["decode_failures"] += 1
                if len(report["examples"]) < max_examples:
                    report["examples"].append(
                        {
                            "sample_key": sample_key,
                            "shard_name": shard_name,
                            "error": str(error),
                        }
                    )

    if errors:
        array = np.asarray(errors, dtype=np.float64)
        report["abs_error_m"] = {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90.0)),
            "p95": float(np.percentile(array, 95.0)),
            "max": float(np.max(array)),
        }
    return report


def write_depth_action_consistency_report(report: dict, out_path: str | Path) -> str:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path.resolve())
