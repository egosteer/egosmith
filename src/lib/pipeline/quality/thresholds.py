"""Auto-resolution of camera-space / episode-motion quality thresholds from a clip-metric set."""

from __future__ import annotations

import numpy as np

from .constants import CAMERA_AXES
from .distribution import summarize_iqr_distribution, summarize_metric_distribution


def _camera_space_bound_metrics(prefix: str, clip_metrics: list[dict], multiplier: float) -> tuple[dict | None, dict]:
    bounds = {}
    distributions = {}
    has_any = False
    for axis in CAMERA_AXES:
        min_key = f"min_camera_space_{prefix}_{axis}"
        max_key = f"max_camera_space_{prefix}_{axis}"
        min_summary = summarize_iqr_distribution([metrics[min_key] for metrics in clip_metrics], multiplier)
        max_summary = summarize_iqr_distribution([metrics[max_key] for metrics in clip_metrics], multiplier)
        if min_summary is None or max_summary is None:
            continue
        has_any = True
        bounds[axis] = {
            "lower": float(min_summary["lower_bound"]),
            "upper": float(max_summary["upper_bound"]),
        }
        distributions[axis] = {
            "lower_tail": min_summary,
            "upper_tail": max_summary,
        }
    return (bounds if has_any else None), distributions


def _camera_space_axis_abs_cap_bounds(cap: float | None) -> dict | None:
    if cap is None:
        return None
    cap_value = float(cap)
    return {
        axis: {
            "lower": -cap_value,
            "upper": cap_value,
        }
        for axis in CAMERA_AXES
    }


def _merge_camera_space_bounds(primary: dict | None, secondary: dict | None) -> dict | None:
    if primary is None:
        return secondary
    if secondary is None:
        return primary
    merged = {}
    for axis in CAMERA_AXES:
        primary_axis = primary.get(axis) if primary else None
        secondary_axis = secondary.get(axis) if secondary else None
        if primary_axis is None and secondary_axis is None:
            continue
        if primary_axis is None:
            merged[axis] = dict(secondary_axis)
            continue
        if secondary_axis is None:
            merged[axis] = dict(primary_axis)
            continue
        merged[axis] = {
            "lower": max(float(primary_axis["lower"]), float(secondary_axis["lower"])),
            "upper": min(float(primary_axis["upper"]), float(secondary_axis["upper"])),
        }
    return merged or None


def resolve_auto_quality_thresholds(clip_metrics: list[dict], criteria: dict) -> dict:
    resolved = {
        "max_camera_space_wrist_abs": criteria["max_camera_space_wrist_abs"],
        "max_camera_space_hand_abs": criteria["max_camera_space_hand_abs"],
        "camera_space_wrist_bounds": None,
        "camera_space_hand_bounds": None,
        "episode_camera_translation_bounds": None,
        "episode_camera_rotation_bounds": None,
    }
    summaries = {}
    auto_method = str(criteria.get("camera_space_auto_method", "iqr_bounds"))
    percentile = float(criteria.get("camera_space_abs_percentile", 99.0))
    scale = float(criteria.get("camera_space_abs_scale", 2.5))
    iqr_multiplier = float(criteria.get("camera_space_iqr_multiplier", 2.5))
    axis_abs_cap = criteria.get("camera_space_axis_abs_cap", 1.5)

    candidate_metrics = [metrics for metrics in clip_metrics if metrics["frames_kept_candidate"] > 0]
    use_manual_abs = (
        resolved["max_camera_space_wrist_abs"] is not None
        or resolved["max_camera_space_hand_abs"] is not None
    )
    if not criteria.get("use_auto_camera_space_thresholds", True) and not use_manual_abs:
        return {
            "resolved": resolved,
            "distribution": summaries,
            "auto_rule": {
                "method": "disabled",
                "percentile": percentile,
                "scale": scale,
                "iqr_multiplier": iqr_multiplier,
                "axis_abs_cap": axis_abs_cap,
            },
        }

    # Episode-level camera-motion IQR (paper Stage-4): dataset distribution of each episode's mean
    # per-frame camera translation/rotation magnitude; flag episodes outside the IQR fence. Runs
    # whenever outlier checks are active, independent of the hand camera-space auto-method.
    episode_iqr_multiplier = float(criteria.get("episode_camera_iqr_multiplier", 2.5))
    cam_trans_summary = summarize_iqr_distribution(
        [metrics.get("mean_camera_translation_step") for metrics in candidate_metrics], episode_iqr_multiplier
    )
    cam_rot_summary = summarize_iqr_distribution(
        [metrics.get("mean_camera_rotation_step") for metrics in candidate_metrics], episode_iqr_multiplier
    )
    if cam_trans_summary is not None:
        resolved["episode_camera_translation_bounds"] = {
            "lower": cam_trans_summary["lower_bound"],
            "upper": cam_trans_summary["upper_bound"],
        }
        summaries["episode_camera_translation"] = cam_trans_summary
    if cam_rot_summary is not None:
        resolved["episode_camera_rotation_bounds"] = {
            "lower": cam_rot_summary["lower_bound"],
            "upper": cam_rot_summary["upper_bound"],
        }
        summaries["episode_camera_rotation"] = cam_rot_summary

    if not use_manual_abs and auto_method == "iqr_bounds":
        wrist_bounds, wrist_distributions = _camera_space_bound_metrics("wrist", candidate_metrics, iqr_multiplier)
        hand_bounds, hand_distributions = _camera_space_bound_metrics("hand", candidate_metrics, iqr_multiplier)
        cap_bounds = _camera_space_axis_abs_cap_bounds(axis_abs_cap)
        resolved["camera_space_wrist_bounds"] = _merge_camera_space_bounds(wrist_bounds, cap_bounds)
        resolved["camera_space_hand_bounds"] = _merge_camera_space_bounds(hand_bounds, cap_bounds)
        if wrist_distributions:
            summaries["camera_space_wrist_bounds"] = wrist_distributions
        if hand_distributions:
            summaries["camera_space_hand_bounds"] = hand_distributions
        if cap_bounds is not None:
            summaries["camera_space_axis_abs_cap"] = {
                "lower": -float(axis_abs_cap),
                "upper": float(axis_abs_cap),
            }
    else:
        wrist_values = [metrics["max_camera_space_wrist_abs"] for metrics in candidate_metrics]
        hand_values = [metrics["max_camera_space_hand_abs"] for metrics in candidate_metrics]
        wrist_summary = summarize_metric_distribution(wrist_values)
        hand_summary = summarize_metric_distribution(hand_values)
        if wrist_summary is not None:
            summaries["max_camera_space_wrist_abs"] = wrist_summary
        if hand_summary is not None:
            summaries["max_camera_space_hand_abs"] = hand_summary

        if resolved["max_camera_space_wrist_abs"] is None and wrist_summary is not None:
            resolved["max_camera_space_wrist_abs"] = float(
                np.percentile(np.asarray(wrist_values, dtype=np.float64), percentile) * scale
            )
        if resolved["max_camera_space_hand_abs"] is None and hand_summary is not None:
            resolved["max_camera_space_hand_abs"] = float(
                np.percentile(np.asarray(hand_values, dtype=np.float64), percentile) * scale
            )

    if use_manual_abs and axis_abs_cap is not None:
        summaries["camera_space_axis_abs_cap"] = {
            "lower": -float(axis_abs_cap),
            "upper": float(axis_abs_cap),
        }

    return {
        "resolved": resolved,
        "distribution": summaries,
        "auto_rule": {
            "method": "manual_abs" if use_manual_abs else auto_method,
            "percentile": percentile,
            "scale": scale,
            "iqr_multiplier": iqr_multiplier,
            "axis_abs_cap": axis_abs_cap,
        },
    }
