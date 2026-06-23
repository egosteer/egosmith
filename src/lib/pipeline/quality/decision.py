"""Final keep/reject decision for one clip, given its metrics and the resolved criteria."""

from __future__ import annotations

from .constants import CAMERA_AXES
from .thresholds import _camera_space_axis_abs_cap_bounds


def _camera_space_bounds_exceeded(metrics: dict, prefix: str, bounds: dict | None) -> bool:
    if not bounds:
        return False
    for axis in CAMERA_AXES:
        axis_bounds = bounds.get(axis)
        if not axis_bounds:
            continue
        min_key = f"min_camera_space_{prefix}_{axis}"
        max_key = f"max_camera_space_{prefix}_{axis}"
        if metrics[min_key] < axis_bounds["lower"] or metrics[max_key] > axis_bounds["upper"]:
            return True
    return False


def decide_clip_quality(
    metrics: dict,
    criteria: dict,
    *,
    include_incomplete_sample_reason: bool = True,
    include_invalid_meta_reason: bool = True,
) -> tuple[bool, list[str]]:
    reasons = []
    if include_incomplete_sample_reason and metrics["incomplete_sample_frames"] > 0:
        reasons.append("incomplete_sample")
    if include_invalid_meta_reason and metrics["invalid_meta_frames"] > 0:
        reasons.append("invalid_meta")
    if metrics["invalid_lowdim_frames"] > 0:
        reasons.append("invalid_lowdim")
    if metrics.get("invalid_rot6d_frames", 0) > 0:
        reasons.append("invalid_rot6d")
    if metrics.get("invalid_extrinsic_frames", 0) > 0:
        reasons.append("invalid_extrinsic")
    if metrics.get("invalid_intrinsic_frames", 0) > 0:
        reasons.append("invalid_intrinsic")
    if metrics["nonfinite_lowdim_frames"] > 0:
        reasons.append("nonfinite_lowdim")
    require_instruction = bool((criteria.get("hard_rules") or {}).get("require_instruction_every_frame", True))
    if require_instruction and metrics.get("missing_instruction_frames", 0) > 0:
        reasons.append("missing_instruction_frame")
    if require_instruction and metrics.get("empty_instruction_frames", 0) > 0:
        reasons.append("empty_instruction_frame")
    if require_instruction and metrics.get("instruction_num_mismatch_frames", 0) > 0:
        reasons.append("instruction_num_mismatch_frame")
    if metrics.get("fatal_visible_left_severe_offscreen_frames", 0) > 0:
        reasons.append("fatal_visible_left_severe_offscreen")
    if metrics.get("fatal_visible_right_severe_offscreen_frames", 0) > 0:
        reasons.append("fatal_visible_right_severe_offscreen")
    if (
        criteria.get("min_visible_hand_any_point_inframe_ratio") is not None
        and metrics.get("visible_left_frames", 0) > 0
        and metrics["visible_left_any_point_inframe_ratio"] < criteria["min_visible_hand_any_point_inframe_ratio"]
    ):
        reasons.append("visible_left_inframe_ratio_below_min")
    if (
        criteria.get("min_visible_hand_any_point_inframe_ratio") is not None
        and metrics.get("visible_right_frames", 0) > 0
        and metrics["visible_right_any_point_inframe_ratio"] < criteria["min_visible_hand_any_point_inframe_ratio"]
    ):
        reasons.append("visible_right_inframe_ratio_below_min")
    if (
        criteria.get("max_visible_hand_all_points_out_of_frame_streak") is not None
        and metrics.get("max_visible_left_out_of_frame_streak", 0) > criteria["max_visible_hand_all_points_out_of_frame_streak"]
    ):
        reasons.append("visible_left_out_of_frame_streak_exceeded")
    if (
        criteria.get("max_visible_hand_all_points_out_of_frame_streak") is not None
        and metrics.get("max_visible_right_out_of_frame_streak", 0) > criteria["max_visible_hand_all_points_out_of_frame_streak"]
    ):
        reasons.append("visible_right_out_of_frame_streak_exceeded")
    if criteria.get("min_instruction_num") is not None and metrics["instruction_num_max"] < criteria["min_instruction_num"]:
        reasons.append("instruction_num_below_min")
    if criteria.get("min_presence_ratio") is not None and metrics["presence_ratio"] < criteria["min_presence_ratio"]:
        reasons.append("presence_ratio_below_min")
    if (
        criteria.get("max_hand_translation_step") is not None
        and metrics["max_hand_translation_step"] > criteria["max_hand_translation_step"]
    ):
        reasons.append("hand_translation_step_exceeded")
    if (
        criteria.get("max_finger_translation_step") is not None
        and metrics.get("max_finger_translation_step", 0.0) > criteria["max_finger_translation_step"]
    ):
        reasons.append("finger_translation_step_exceeded")
    if (
        criteria.get("max_camera_translation_step") is not None
        and metrics["max_camera_translation_step"] > criteria["max_camera_translation_step"]
    ):
        reasons.append("camera_translation_step_exceeded")
    if (
        criteria.get("max_camera_rotation_step") is not None
        and metrics["max_camera_rotation_step"] > criteria["max_camera_rotation_step"]
    ):
        reasons.append("camera_rotation_step_exceeded")
    if (
        criteria.get("max_wrist_rotation_step") is not None
        and metrics.get("max_wrist_rotation_step", 0.0) > criteria["max_wrist_rotation_step"]
    ):
        reasons.append("wrist_rotation_step_exceeded")
    if (
        criteria.get("max_camera_space_wrist_abs") is not None
        and metrics["max_camera_space_wrist_abs"] > criteria["max_camera_space_wrist_abs"]
    ):
        reasons.append("camera_space_wrist_abs_exceeded")
    if (
        criteria.get("max_camera_space_hand_abs") is not None
        and metrics["max_camera_space_hand_abs"] > criteria["max_camera_space_hand_abs"]
    ):
        reasons.append("camera_space_hand_abs_exceeded")
    if _camera_space_bounds_exceeded(metrics, "wrist", criteria.get("camera_space_wrist_bounds")):
        reasons.append("camera_space_wrist_iqr_bounds_exceeded")
    if _camera_space_bounds_exceeded(metrics, "hand", criteria.get("camera_space_hand_bounds")):
        reasons.append("camera_space_hand_iqr_bounds_exceeded")
    axis_abs_cap = criteria.get("camera_space_axis_abs_cap")
    if axis_abs_cap is not None:
        cap_bounds = _camera_space_axis_abs_cap_bounds(axis_abs_cap)
        if (
            criteria.get("camera_space_wrist_bounds") is None
            and _camera_space_bounds_exceeded(metrics, "wrist", cap_bounds)
        ):
            reasons.append("camera_space_wrist_axis_abs_cap_exceeded")
        if (
            criteria.get("camera_space_hand_bounds") is None
            and _camera_space_bounds_exceeded(metrics, "hand", cap_bounds)
        ):
            reasons.append("camera_space_hand_axis_abs_cap_exceeded")
    episode_translation_bounds = criteria.get("episode_camera_translation_bounds")
    if episode_translation_bounds is not None:
        value = float(metrics.get("mean_camera_translation_step", 0.0))
        if value < episode_translation_bounds["lower"] or value > episode_translation_bounds["upper"]:
            reasons.append("episode_camera_translation_iqr_exceeded")
    episode_rotation_bounds = criteria.get("episode_camera_rotation_bounds")
    if episode_rotation_bounds is not None:
        value = float(metrics.get("mean_camera_rotation_step", 0.0))
        if value < episode_rotation_bounds["lower"] or value > episode_rotation_bounds["upper"]:
            reasons.append("episode_camera_rotation_iqr_exceeded")
    return not reasons, reasons
