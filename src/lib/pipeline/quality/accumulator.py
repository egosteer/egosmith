"""Per-clip quality-stats accumulation lifecycle: init -> per-frame update -> finalize."""

from __future__ import annotations

import numpy as np

from .constants import LEFT_ROOT_ROT6D_SLICE, LOWDIM_SIZE, RIGHT_ROOT_ROT6D_SLICE
from .kinematics import (
    _abs_axis_metrics,
    _fingers_relative_to_wrist,
    camera_space_abs_metrics,
    camera_space_axis_metrics,
    windowed_wrist_camera_extremes,
)
from .lowdim import (
    _rot6d_to_rotmat,
    extract_lowdim_components,
    is_finite_array,
    validate_lowdim_numeric_sanity,
)
from .projection import classify_hand_projection


def new_clip_quality_stats(
    clip_id: str,
    *,
    target_fps: float = 30.0,
    chunk_window_past_seconds: float = 6.0,
    chunk_window_future_frames: int = 30,
    enable_chunk_window: bool = False,
) -> dict:
    return {
        "clip_id": clip_id,
        "frames_total": 0,
        "frames_kept_candidate": 0,
        "presence_nonzero_frames": 0,
        "incomplete_sample_frames": 0,
        "nonfinite_lowdim_frames": 0,
        "invalid_meta_frames": 0,
        "invalid_lowdim_frames": 0,
        "invalid_rot6d_frames": 0,
        "invalid_extrinsic_frames": 0,
        "invalid_intrinsic_frames": 0,
        "missing_instruction_frames": 0,
        "empty_instruction_frames": 0,
        "instruction_num_mismatch_frames": 0,
        "instruction_num_max": 0,
        "max_hand_translation_step": 0.0,
        "max_finger_translation_step": 0.0,
        "max_camera_translation_step": 0.0,
        "max_camera_rotation_step": 0.0,
        "max_wrist_rotation_step": 0.0,
        "_camera_translation_step_sum": 0.0,
        "_camera_rotation_step_sum": 0.0,
        "_camera_step_count": 0,
        "max_camera_space_wrist_abs": 0.0,
        "max_camera_space_hand_abs": 0.0,
        "visible_left_frames": 0,
        "visible_right_frames": 0,
        "visible_left_any_point_inframe_frames": 0,
        "visible_right_any_point_inframe_frames": 0,
        "visible_left_all_points_out_of_frame_frames": 0,
        "visible_right_all_points_out_of_frame_frames": 0,
        "fatal_visible_left_severe_offscreen_frames": 0,
        "fatal_visible_right_severe_offscreen_frames": 0,
        "max_visible_left_out_of_frame_streak": 0,
        "max_visible_right_out_of_frame_streak": 0,
        "_camera_space_wrist_min": np.full((3,), np.inf, dtype=np.float32),
        "_camera_space_wrist_max": np.full((3,), -np.inf, dtype=np.float32),
        "_camera_space_hand_min": np.full((3,), np.inf, dtype=np.float32),
        "_camera_space_hand_max": np.full((3,), -np.inf, dtype=np.float32),
        "_chunk_window_enabled": bool(enable_chunk_window),
        "_chunk_window_past_frames": int(round(float(chunk_window_past_seconds) * float(target_fps))),
        "_chunk_window_future_frames": int(chunk_window_future_frames),
        "_seq_frame_idx": [],
        "_seq_wrist_world": [],
        "_seq_extrinsic": [],
        "_visible_left_out_of_frame_streak": 0,
        "_visible_right_out_of_frame_streak": 0,
        "_prev_frame_idx": None,
        "_prev_left": None,
        "_prev_right": None,
        "_prev_left_fingers": None,
        "_prev_right_fingers": None,
        "_prev_left_rot": None,
        "_prev_right_rot": None,
        "_prev_extrinsic": None,
        "_prev_finite": False,
    }


def update_clip_quality_stats(
    stats: dict,
    frame_idx: int,
    instruction_num: int,
    presence: int,
    lowdim,
    *,
    missing_instruction: bool = False,
    empty_instruction: bool = False,
    instruction_num_mismatch: bool = False,
    count_invalid_lowdim: bool = True,
    compute_motion_metrics: bool = True,
    compute_camera_space_metrics: bool = True,
    image_size: tuple[int, int] | None = None,
    severe_offscreen_scale: float = 1.4,
) -> None:
    stats["frames_total"] += 1
    stats["instruction_num_max"] = max(stats["instruction_num_max"], int(instruction_num))
    if missing_instruction:
        stats["missing_instruction_frames"] += 1
    if empty_instruction:
        stats["empty_instruction_frames"] += 1
    if instruction_num_mismatch:
        stats["instruction_num_mismatch_frames"] += 1
    if int(presence) > 0:
        stats["presence_nonzero_frames"] += 1

    if lowdim is None:
        if count_invalid_lowdim:
            stats["invalid_lowdim_frames"] += 1
        stats["_prev_frame_idx"] = int(frame_idx)
        stats["_prev_left"] = None
        stats["_prev_right"] = None
        stats["_prev_extrinsic"] = None
        stats["_prev_finite"] = False
        return

    try:
        lowdim_array = np.asarray(lowdim, dtype=np.float32).reshape(-1)
    except Exception:
        lowdim_array = None
    if lowdim_array is None or lowdim_array.shape != (LOWDIM_SIZE,):
        if count_invalid_lowdim:
            stats["invalid_lowdim_frames"] += 1
        stats["_prev_frame_idx"] = int(frame_idx)
        stats["_prev_left"] = None
        stats["_prev_right"] = None
        stats["_prev_extrinsic"] = None
        stats["_prev_finite"] = False
        return

    if not is_finite_array(lowdim_array):
        stats["nonfinite_lowdim_frames"] += 1
        stats["_prev_frame_idx"] = int(frame_idx)
        stats["_prev_left"] = None
        stats["_prev_right"] = None
        stats["_prev_extrinsic"] = None
        stats["_prev_finite"] = False
        return

    sanity = validate_lowdim_numeric_sanity(lowdim_array)
    if not sanity["valid"]:
        if count_invalid_lowdim:
            stats["invalid_lowdim_frames"] += 1
        if sanity["invalid_rot6d"]:
            stats["invalid_rot6d_frames"] += 1
        if sanity["invalid_extrinsic"]:
            stats["invalid_extrinsic_frames"] += 1
        if sanity["invalid_intrinsic"]:
            stats["invalid_intrinsic_frames"] += 1
        stats["_prev_frame_idx"] = int(frame_idx)
        stats["_prev_left"] = None
        stats["_prev_right"] = None
        stats["_prev_extrinsic"] = None
        stats["_prev_finite"] = False
        return

    stats["frames_kept_candidate"] += 1
    if not compute_motion_metrics and not compute_camera_space_metrics:
        stats["_prev_frame_idx"] = None
        stats["_prev_left"] = None
        stats["_prev_right"] = None
        stats["_prev_extrinsic"] = None
        stats["_prev_finite"] = False
        return

    parts = extract_lowdim_components(lowdim_array)
    current_left = parts["left_translation"]
    current_right = parts["right_translation"]
    left_fingertips = parts["left_fingertips"]
    right_fingertips = parts["right_fingertips"]
    current_extrinsic = parts["extrinsic"]
    intrinsic = parts["intrinsic"]

    if compute_camera_space_metrics:
        wrist_camera_metrics = camera_space_abs_metrics(
            np.stack([current_left, current_right], axis=0),
            current_extrinsic,
        )
        wrist_axis_metrics = camera_space_axis_metrics(
            np.stack([current_left, current_right], axis=0),
            current_extrinsic,
        )
        if stats["_chunk_window_enabled"]:
            # Buffer the per-frame wrist (world) + camera extrinsic so finalize can compute the
            # sliding-window wrist-relative-to-camera extremes (paper Stage-4 chunk level).
            stats["_seq_frame_idx"].append(int(frame_idx))
            stats["_seq_wrist_world"].append(
                np.stack([current_left, current_right], axis=0).astype(np.float32)
            )
            stats["_seq_extrinsic"].append(np.asarray(current_extrinsic, dtype=np.float32).reshape(4, 4))
        # Finger joints relative to the wrist (paper Stage-4 chunk level), not relative to camera.
        left_finger_wrist = _fingers_relative_to_wrist(
            left_fingertips, current_left, lowdim_array[LEFT_ROOT_ROT6D_SLICE]
        )
        right_finger_wrist = _fingers_relative_to_wrist(
            right_fingertips, current_right, lowdim_array[RIGHT_ROOT_ROT6D_SLICE]
        )
        hand_camera_metrics, hand_axis_metrics = _abs_axis_metrics(
            np.concatenate([left_finger_wrist, right_finger_wrist], axis=0)
        )
        stats["max_camera_space_wrist_abs"] = max(
            stats["max_camera_space_wrist_abs"],
            wrist_camera_metrics["max_abs"],
        )
        stats["max_camera_space_hand_abs"] = max(
            stats["max_camera_space_hand_abs"],
            hand_camera_metrics["max_abs"],
        )
        stats["_camera_space_wrist_min"] = np.minimum(
            stats["_camera_space_wrist_min"],
            np.asarray([wrist_axis_metrics["min_x"], wrist_axis_metrics["min_y"], wrist_axis_metrics["min_z"]], dtype=np.float32),
        )
        stats["_camera_space_wrist_max"] = np.maximum(
            stats["_camera_space_wrist_max"],
            np.asarray([wrist_axis_metrics["max_x"], wrist_axis_metrics["max_y"], wrist_axis_metrics["max_z"]], dtype=np.float32),
        )
        stats["_camera_space_hand_min"] = np.minimum(
            stats["_camera_space_hand_min"],
            np.asarray([hand_axis_metrics["min_x"], hand_axis_metrics["min_y"], hand_axis_metrics["min_z"]], dtype=np.float32),
        )
        stats["_camera_space_hand_max"] = np.maximum(
            stats["_camera_space_hand_max"],
            np.asarray([hand_axis_metrics["max_x"], hand_axis_metrics["max_y"], hand_axis_metrics["max_z"]], dtype=np.float32),
        )

    if image_size is not None:
        left_visible = bool(int(presence) & 1)
        right_visible = bool(int(presence) & 2)
        hand_projection_specs = (
            (
                "left",
                left_visible,
                np.concatenate([current_left.reshape(1, 3), left_fingertips], axis=0),
            ),
            (
                "right",
                right_visible,
                np.concatenate([current_right.reshape(1, 3), right_fingertips], axis=0),
            ),
        )
        for hand_name, is_visible, points_world in hand_projection_specs:
            if not is_visible:
                stats[f"_visible_{hand_name}_out_of_frame_streak"] = 0
                continue
            stats[f"visible_{hand_name}_frames"] += 1
            projection = classify_hand_projection(
                points_world,
                current_extrinsic,
                intrinsic,
                image_size,
                severe_offscreen_scale=severe_offscreen_scale,
            )
            if projection["any_point_inframe"]:
                stats[f"visible_{hand_name}_any_point_inframe_frames"] += 1
                stats[f"_visible_{hand_name}_out_of_frame_streak"] = 0
            else:
                stats[f"_visible_{hand_name}_out_of_frame_streak"] += 1
                stats[f"max_visible_{hand_name}_out_of_frame_streak"] = max(
                    stats[f"max_visible_{hand_name}_out_of_frame_streak"],
                    stats[f"_visible_{hand_name}_out_of_frame_streak"],
                )
            if projection["all_points_out_of_frame"]:
                stats[f"visible_{hand_name}_all_points_out_of_frame_frames"] += 1
            if projection["all_points_severe_offscreen"]:
                stats[f"fatal_visible_{hand_name}_severe_offscreen_frames"] += 1

    if compute_motion_metrics:
        prev_idx = stats["_prev_frame_idx"]
        curr_left_rot = _rot6d_to_rotmat(lowdim_array[LEFT_ROOT_ROT6D_SLICE])
        curr_right_rot = _rot6d_to_rotmat(lowdim_array[RIGHT_ROOT_ROT6D_SLICE])
        if stats["_prev_finite"] and prev_idx is not None:
            frame_gap = max(1, int(frame_idx) - int(prev_idx))
            left_step = float(np.linalg.norm(current_left - stats["_prev_left"]) / frame_gap)
            right_step = float(np.linalg.norm(current_right - stats["_prev_right"]) / frame_gap)
            # Per-fingertip displacement; take the largest fingertip step across both hands.
            left_finger_step = float(
                np.linalg.norm(left_fingertips - stats["_prev_left_fingers"], axis=1).max() / frame_gap
            )
            right_finger_step = float(
                np.linalg.norm(right_fingertips - stats["_prev_right_fingers"], axis=1).max() / frame_gap
            )
            prev_rot = stats["_prev_extrinsic"][:3, :3]
            prev_trans = stats["_prev_extrinsic"][:3, 3]
            curr_rot = current_extrinsic[:3, :3]
            curr_trans = current_extrinsic[:3, 3]
            camera_translation_step = float(np.linalg.norm(curr_trans - prev_trans) / frame_gap)
            camera_rotation_step = float(np.linalg.norm((curr_rot - prev_rot).reshape(-1)) / frame_gap)
            # Wrist (root) rotation step: Frobenius norm of the per-frame root-rotation delta, max
            # over both hands. Same metric family as camera_rotation_step, so the threshold maps the
            # same way (||R1-R2||_F = 2*sqrt(2)*sin(theta/2)); paper cap is 41 deg/frame (~0.99).
            left_wrist_rotation_step = float(
                np.linalg.norm((curr_left_rot - stats["_prev_left_rot"]).reshape(-1)) / frame_gap
            )
            right_wrist_rotation_step = float(
                np.linalg.norm((curr_right_rot - stats["_prev_right_rot"]).reshape(-1)) / frame_gap
            )

            stats["max_hand_translation_step"] = max(
                stats["max_hand_translation_step"],
                left_step,
                right_step,
            )
            stats["max_finger_translation_step"] = max(
                stats["max_finger_translation_step"],
                left_finger_step,
                right_finger_step,
            )
            stats["max_camera_translation_step"] = max(
                stats["max_camera_translation_step"],
                camera_translation_step,
            )
            stats["max_camera_rotation_step"] = max(
                stats["max_camera_rotation_step"],
                camera_rotation_step,
            )
            stats["max_wrist_rotation_step"] = max(
                stats["max_wrist_rotation_step"],
                left_wrist_rotation_step,
                right_wrist_rotation_step,
            )
            # Episode-level camera-motion accumulators (mean per-frame magnitude -> dataset IQR).
            stats["_camera_translation_step_sum"] += camera_translation_step
            stats["_camera_rotation_step_sum"] += camera_rotation_step
            stats["_camera_step_count"] += 1

        stats["_prev_frame_idx"] = int(frame_idx)
        stats["_prev_left"] = current_left
        stats["_prev_right"] = current_right
        stats["_prev_left_fingers"] = left_fingertips
        stats["_prev_right_fingers"] = right_fingertips
        stats["_prev_left_rot"] = curr_left_rot
        stats["_prev_right_rot"] = curr_right_rot
        stats["_prev_extrinsic"] = current_extrinsic
        stats["_prev_finite"] = True
    else:
        stats["_prev_frame_idx"] = None
        stats["_prev_left"] = None
        stats["_prev_right"] = None
        stats["_prev_left_fingers"] = None
        stats["_prev_right_fingers"] = None
        stats["_prev_left_rot"] = None
        stats["_prev_right_rot"] = None
        stats["_prev_extrinsic"] = None
        stats["_prev_finite"] = False


def finalize_clip_quality_metrics(stats: dict) -> dict:
    def _axis_value(array, axis_idx: int, *, fallback: float) -> float:
        value = float(array[axis_idx])
        return fallback if not np.isfinite(value) else value

    metrics = {
        "frames_total": int(stats["frames_total"]),
        "frames_kept_candidate": int(stats["frames_kept_candidate"]),
        "presence_ratio": (
            float(stats["presence_nonzero_frames"]) / float(stats["frames_total"])
            if stats["frames_total"] > 0
            else 0.0
        ),
        "instruction_num_max": int(stats["instruction_num_max"]),
        "incomplete_sample_frames": int(stats["incomplete_sample_frames"]),
        "nonfinite_lowdim_frames": int(stats["nonfinite_lowdim_frames"]),
        "invalid_meta_frames": int(stats["invalid_meta_frames"]),
        "invalid_lowdim_frames": int(stats["invalid_lowdim_frames"]),
        "invalid_rot6d_frames": int(stats["invalid_rot6d_frames"]),
        "invalid_extrinsic_frames": int(stats["invalid_extrinsic_frames"]),
        "invalid_intrinsic_frames": int(stats["invalid_intrinsic_frames"]),
        "missing_instruction_frames": int(stats["missing_instruction_frames"]),
        "empty_instruction_frames": int(stats["empty_instruction_frames"]),
        "instruction_num_mismatch_frames": int(stats["instruction_num_mismatch_frames"]),
        "max_hand_translation_step": float(stats["max_hand_translation_step"]),
        "max_finger_translation_step": float(stats["max_finger_translation_step"]),
        "max_camera_translation_step": float(stats["max_camera_translation_step"]),
        "max_camera_rotation_step": float(stats["max_camera_rotation_step"]),
        "max_wrist_rotation_step": float(stats["max_wrist_rotation_step"]),
        "mean_camera_translation_step": (
            float(stats["_camera_translation_step_sum"]) / float(stats["_camera_step_count"])
            if stats["_camera_step_count"] > 0
            else 0.0
        ),
        "mean_camera_rotation_step": (
            float(stats["_camera_rotation_step_sum"]) / float(stats["_camera_step_count"])
            if stats["_camera_step_count"] > 0
            else 0.0
        ),
        "max_camera_space_wrist_abs": float(stats["max_camera_space_wrist_abs"]),
        "max_camera_space_hand_abs": float(stats["max_camera_space_hand_abs"]),
        "visible_left_frames": int(stats["visible_left_frames"]),
        "visible_right_frames": int(stats["visible_right_frames"]),
        "visible_left_any_point_inframe_ratio": (
            float(stats["visible_left_any_point_inframe_frames"]) / float(stats["visible_left_frames"])
            if stats["visible_left_frames"] > 0
            else 1.0
        ),
        "visible_right_any_point_inframe_ratio": (
            float(stats["visible_right_any_point_inframe_frames"]) / float(stats["visible_right_frames"])
            if stats["visible_right_frames"] > 0
            else 1.0
        ),
        "visible_left_all_points_out_of_frame_ratio": (
            float(stats["visible_left_all_points_out_of_frame_frames"]) / float(stats["visible_left_frames"])
            if stats["visible_left_frames"] > 0
            else 0.0
        ),
        "visible_right_all_points_out_of_frame_ratio": (
            float(stats["visible_right_all_points_out_of_frame_frames"]) / float(stats["visible_right_frames"])
            if stats["visible_right_frames"] > 0
            else 0.0
        ),
        "fatal_visible_left_severe_offscreen_frames": int(stats["fatal_visible_left_severe_offscreen_frames"]),
        "fatal_visible_right_severe_offscreen_frames": int(stats["fatal_visible_right_severe_offscreen_frames"]),
        "max_visible_left_out_of_frame_streak": int(stats["max_visible_left_out_of_frame_streak"]),
        "max_visible_right_out_of_frame_streak": int(stats["max_visible_right_out_of_frame_streak"]),
        "min_camera_space_wrist_x": _axis_value(stats["_camera_space_wrist_min"], 0, fallback=0.0),
        "max_camera_space_wrist_x": _axis_value(stats["_camera_space_wrist_max"], 0, fallback=0.0),
        "min_camera_space_wrist_y": _axis_value(stats["_camera_space_wrist_min"], 1, fallback=0.0),
        "max_camera_space_wrist_y": _axis_value(stats["_camera_space_wrist_max"], 1, fallback=0.0),
        "min_camera_space_wrist_z": _axis_value(stats["_camera_space_wrist_min"], 2, fallback=0.0),
        "max_camera_space_wrist_z": _axis_value(stats["_camera_space_wrist_max"], 2, fallback=0.0),
        "min_camera_space_hand_x": _axis_value(stats["_camera_space_hand_min"], 0, fallback=0.0),
        "max_camera_space_hand_x": _axis_value(stats["_camera_space_hand_max"], 0, fallback=0.0),
        "min_camera_space_hand_y": _axis_value(stats["_camera_space_hand_min"], 1, fallback=0.0),
        "max_camera_space_hand_y": _axis_value(stats["_camera_space_hand_max"], 1, fallback=0.0),
        "min_camera_space_hand_z": _axis_value(stats["_camera_space_hand_min"], 2, fallback=0.0),
        "max_camera_space_hand_z": _axis_value(stats["_camera_space_hand_max"], 2, fallback=0.0),
    }
    # Override the per-frame wrist (relative-to-camera) extremes with the sliding-window extremes
    # (paper Stage-4 chunk level). Falls back to the per-frame values when windowing is disabled or
    # no frames were buffered. Finger (relative-to-wrist) extremes are frame-local, already final.
    if stats.get("_chunk_window_enabled") and stats.get("_seq_frame_idx"):
        axis_min, axis_max, max_abs = windowed_wrist_camera_extremes(
            stats["_seq_frame_idx"],
            stats["_seq_wrist_world"],
            stats["_seq_extrinsic"],
            past_frames=stats["_chunk_window_past_frames"],
            future_frames=stats["_chunk_window_future_frames"],
        )
        metrics["max_camera_space_wrist_abs"] = float(max_abs)
        metrics["min_camera_space_wrist_x"] = float(axis_min[0])
        metrics["max_camera_space_wrist_x"] = float(axis_max[0])
        metrics["min_camera_space_wrist_y"] = float(axis_min[1])
        metrics["max_camera_space_wrist_y"] = float(axis_max[1])
        metrics["min_camera_space_wrist_z"] = float(axis_min[2])
        metrics["max_camera_space_wrist_z"] = float(axis_max[2])
    return metrics
